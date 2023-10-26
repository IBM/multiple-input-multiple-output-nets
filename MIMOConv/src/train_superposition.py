#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
from models.superwideisonet import *
from models.superwideresnet import *
from datasets import *
from torch.utils import tensorboard, data
from optimizer import construct_optim
from mixup import *
import argparse
import sys
import random, numpy
from math import isfinite
import tqdm

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
def train_epoch(average_gradient_last_epoch):
    r"""trains model for one epoch
        Args:
            average_gradient_last_epoch: average gradient of last epoch. Determines which batches to discard
        Returns:
            tuple including loss, accuracy, average gradient norm and fraction of filtered batches
        """
    sum_loss = 0.0
    correct = 0
    total = 0
    isometry_regularization_grad_norm_sum = 0
    total_grad_norm_sum = 0
    non_finite_gradient_count = 0
    finite_gradient_count = 0

    # set a standard value if validation changed it
    model.num_img_sup = args.num 
    for _ in range(args.num): # batch size scales proportionally with args.num. Must correct for fewer updates to be fair.
        for inputs, labels in trainloader: 
            if args.sup_frequency < 1:
                if torch.rand(1) > args.sup_frequency:
                    model.num_img_sup = sup_low # batch size in bulk of the model after binding increases by factor args.num / sup_low
                else:
                    model.num_img_sup = args.num

            # transfer data to GPU
            inputs = inputs.to(device)
            labels = labels.to(device) 

            # mixup data augmentation
            inputs, labels_a, labels_b, lambd = mixup_data(inputs, labels, args.mixup_alpha, cuda_available)

            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            effective_batch_size = outputs.shape[0] # due to superposition batch may be truncated
            labels_a = labels_a[:effective_batch_size]
            labels_b = labels_b[:effective_batch_size]

            # backward pass
            regularization_loss = args.orthogonal_regularization * model.isometry_regularization(device)
            if regularization_loss != 0: # WideResNet has no regularization loss
                regularization_loss.backward()
                isometry_regularization_grad_norm_sum += torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3) # used to measure norm

            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lambd) + args.binding_regularization * model.binding_regularization()
            loss.backward()
            # allows at most twice the total gradient average of last epoch. Prevents model divergence by filtering out bad batches.
            # if learning rate is chosen too high and all batches exceed twice the old mean in gradient norm model stops updating
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1e3) #used to measure norm

            # update
            if total_grad_norm.isfinite and total_grad_norm < args.gradient_clipping * average_gradient_last_epoch:
                optimizer.step()
                total_grad_norm_sum += total_grad_norm
                finite_gradient_count += 1
            else:
                print("batch produced exceedingly large gradient - no update made")
                non_finite_gradient_count += 1

            # statistics
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += effective_batch_size
            # mixup estimate of correctness: although a superposition is given, it can only estimate a single class, hence reported average will be naturally low
            correct += lambd * (predicted == labels_a).sum().item() + (1 - lambd) * (predicted == labels_b).sum().item()
            scheduler.step()
    accuracy = 100 * correct / total
    loss = sum_loss / (total / effective_batch_size)

    return loss, accuracy, (isometry_regularization_grad_norm_sum / finite_gradient_count if finite_gradient_count > 0 else -1), (total_grad_norm_sum / finite_gradient_count if finite_gradient_count > 0 else average_gradient_last_epoch), (non_finite_gradient_count / (non_finite_gradient_count + finite_gradient_count))

def validate_epoch():
    r"""validates model at a given epoch
        Returns:
            tuple of loss and accuracy
        """
    sum_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in evalloader:
            # transfer data to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            if args.dataset=="FLOWERS":
                labels = labels -1
            # calculate outputs by running images through the network
            outputs = model(inputs)
            effective_batch_size = outputs.shape[0] # due to superposition batch may be truncated
            labels = labels[:effective_batch_size]
            loss = criterion(outputs, labels)
            
            # statistics
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += effective_batch_size
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        loss = sum_loss / (total / effective_batch_size)

        return loss, accuracy

if __name__ == '__main__': # avoids rerunning code when multiple processes are spawned (for quicker dataloading)

    #------------- argument parsing --------------- 
    # only -y, -m not used
    parser = argparse.ArgumentParser(description='Trains and evaluates CNNs, in particular demonstrating superposition principles', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', type=str, choices=["WideResNet-28", "WideISOReLUNet-28", "WideISONet-28", "MIMONet-28", "WideResNet-16", "WideISOReLUNet-16", 
                                                    "WideISONet-16", "MIMONet-16", "MIMONet-10"], help='architecture and network depth')
    parser.add_argument('dataset', type=str, choices=["CIFAR10", "CIFAR100", "MNIST", "SVHN"], help='dataset')
    parser.add_argument("type", type=str, choices=["None", "HRR", "MBAT"], help="binding type")
    parser.add_argument("num", type=int, help="maximum superposition capability of model")
    parser.add_argument("-x", "--experiment_nr", type=int, required=True, help="useful for storing results in separate folders")
    parser.add_argument("-f", "--final", action="store_true", help="run on train/test split instead of train/val split - should only be enabled at the very end of hypertuning")

    parser.add_argument("-w", "--width", type=int, default=1, help="width of network, i.e. factor to increase channel size")
    parser.add_argument("-i", "--initial_width", type=int, default=1, help="width of output of initial convolutional layer, which is not affected by [width]")
    parser.add_argument('-p', "--relu_parameter_init", type=float, default=None, help="offset in shiftedReLU (default -1) or parameter in parametricReLU (default 0.5) depending on the model")
    parser.add_argument("-k", "--skip_init", action="store_true", help="whether skip_init (ignore residual branch at initialisation) is enabled")
    parser.add_argument("-q", "--dirac_init", action="store_true", help="whether dirac initialisation of convolutions is enabled")
    parser.add_argument("-z", "--batch_norm_disabled", action="store_true", help="whether batch norm is disabled")
    parser.add_argument("-v", "--trainable_keys_disabled", action="store_true", help="whether keys are fixed after initialisation")

    parser.add_argument("-j", "--binding_regularization", type=float, default=0.1, help="regularization factor to produce orthogonal unit norm HRR vectors")
    parser.add_argument("-o", "--orthogonal_regularization", type=float, default=1e-4, help="regularization factor keeping convolutions isometric in MIMONet and WideISONet")

    parser.add_argument('-u', "--sup_frequency", type=float, default=1.0, help="fraction of training batches with [num] images are superposed (instead of [sup_low]) should be between 0 and 1")
    parser.add_argument('-s', "--sup_low", type=int, default=None, help="how many images are superposed in the low-demand setting. If left unspecified always [num] images are superposed")

    parser.add_argument("-b", "--batch_size", type=int, default=128, help="the batch size used. The batch size before binding is larger by a factor [num]")
    parser.add_argument("-n", "--number_of_cpus", type=int, default=0, help="number of cpus used in dataloading")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="the number of epochs during training. Note that for a fair comparison each epoch passes through data [num] times")
    parser.add_argument("-l", "--lr", type=float, default=0.2, help="the maximal learning rate (oneCycleLR Policy)")
    parser.add_argument("-d", "--weight_decay", type=float, default=1e-5, help="weight_decay, i.e. l2 regularization. Not applied to relu parameters")
    parser.add_argument("-a", "--mixup_alpha", type=float, default=1.0, help="high alpha yields strong mixup, alpha = 1 uniform mixup, alpha = 0 no mixup")
    parser.add_argument("-g", "--gradient_clipping", type=float, default=10.0, help="the cutoff ratio (compared to average gradient norm of last epoch) selecting which batches are filtered out")
    parser.add_argument("-c", "--checkpointing_path", type=str, default="results/", help="saves model to path after each epoch")
    parser.add_argument('-t', "--transfer_learning_path", type=str, default=None, help="loads model from checkpoint at path as starting point")
    parser.add_argument('-r', "--random_seed", type=int, default=42, help="allows reproducibility")
 
    args = parser.parse_args()

    identifier = f'Experiment{args.experiment_nr}/{args.model}/{args.dataset}/{args.type}/{args.num}/{args.width}/{args.initial_width}/{args.relu_parameter_init}/{args.skip_init}/{args.dirac_init}/{args.batch_norm_disabled}/{args.binding_regularization}/{args.orthogonal_regularization}/{args.sup_frequency}/{args.sup_low}/{args.batch_size}/{args.number_of_cpus}/{args.epochs}/{args.lr}/{args.weight_decay}/{args.mixup_alpha}/{args.gradient_clipping}/{args.random_seed}'
    identifier_ = identifier.replace('/', '_')

     #------------- reproducibility ---------------  
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # seed dataloader workers
    def seed_worker(worker_id):
        numpy.random.seed(args.random_seed)
        random.seed(args.random_seed)
    # seed generators
    g = torch.Generator()
    g.manual_seed(args.random_seed)

    #------------- logging ---------------  
    description_dict = {
        'experiment' : args.experiment_nr,
        'model': args.model,
        'dataset': args.dataset,
        'binding_type': args.type,
        'superposition_capacity': args.num,
        'width': args.width,
        'initial_width': args.initial_width,
        'relu_parameter_init': args.relu_parameter_init,
        'skip_init': args.skip_init,
        'dirac_init': args.dirac_init,
        'batch_norm_disabled': args.batch_norm_disabled,
        'trainable_keys_disabled' : args.trainable_keys_disabled,
        'binding_regularization_coefficient': args.binding_regularization,
        'orthogonal_regularization_coefficient': args.orthogonal_regularization,
        'superposition_frequency': args.sup_frequency,
        'low_superposition_number': args.sup_low,
        'batch_size': args.batch_size,
        'cpus' : args.number_of_cpus,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'mixup_alpha': args.mixup_alpha,
        'gradient_clipping' : args.gradient_clipping,
        'seed' : args.random_seed,
    }

    writer = tensorboard.SummaryWriter(log_dir= 'runs/' + identifier)

    #------------- process settings ---------------
    sup_low = None
    if args.sup_low:
        if args.sup_low > args.num or args.sup_low < 1 or args.num % args.sup_low != 0: 
            print(f'option --sup_low (-s) must divide argument num')
            sys.exit(2)
        else:
            sup_low = args.sup_low
    else:
        sup_low = args.num

    if args.dataset == "CIFAR10" :
        num_classes = 10
        input_channels = 3
    elif args.dataset == "MNIST":
        num_classes = 10
        input_channels = 1
    elif args.dataset == "CIFAR100":
        num_classes = 100
        input_channels = 3
    elif args.dataset == "SVHN":
        num_classes = 10 
        input_channels = 3
    else:
        print(f'unknown argument {args.dataset} for dataset')
        sys.exit(2)

    if args.batch_norm_disabled:
        norm = IdentityNorm
    else:
        norm = None # defaults to BatchNorm

    model = {"WideResNet-16":SuperWideResnet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideISONet-16":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideISOReLUNet-16":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMONet-10":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [1, 1, 1], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMONet-16":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideResNet-28":SuperWideResnet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideISONet-28":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideISOReLUNet-28":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMONet-28":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels)
             }.get(args.model)
    if model == None:
        print(f'unknown argument {args.model} for model')
        sys.exit(2)

    #------------- finished parsing arguments and fixing settings ---------------

    if args.transfer_learning_path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.transfer_learning_path)['model_state_dict']
        pretrained_dict = { k:v for k,v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size() }
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # only keep weights that exist in new model (allows loading models with different capacity)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded from path {:}".format(args.transfer_learning_path))

    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    pin_memory = True if cuda_available else False # pin memory may speed up training by allowing faster data loading from dedicated main memory, depends on system used.

    trainset, evalset = get_train_eval_test_sets(args.dataset, args.final)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size*args.num, # batchsize increase supposed to counteract the division of the batch at superposition, i.e. preserve its size for most of the model
                                    shuffle=True, num_workers=args.number_of_cpus, pin_memory=pin_memory, drop_last=True, worker_init_fn=seed_worker, generator=g) 
    train_iters = len(trainloader)
    evalloader = data.DataLoader(evalset, batch_size=args.batch_size*args.num,
                                    shuffle=True, num_workers=args.number_of_cpus, pin_memory=pin_memory, drop_last=False, worker_init_fn=seed_worker, generator=g)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = construct_optim(model, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, steps_per_epoch=train_iters*args.num)

    try:
        checkpoint = torch.load(args.checkpointing_path + identifier_ + '_model.pt')
        writer.add_text("WARNING", f"Loading checkpoint at epoch {checkpoint['epoch']}")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    except Exception as e: 
        print(e)
        writer.add_text("Exceptions", str(e))
        checkpoint = {
                    'model_state_dict': None,
                    'optimizer_state_dict': None,
                    'scheduler_state_dict': None,
                    'epoch' : -1,
                }        

    #------------- Training Loop ---------------
    average_gradient_last_epoch = float('inf')
    for epoch in tqdm.tqdm(range(args.epochs)):
        if checkpoint['epoch'] < epoch:

            # train
            loss_train, accuracy_train, isometry_regularization_grad_norm, average_gradient_last_epoch, non_finite_gradient_fraction = train_epoch(average_gradient_last_epoch)

            if isfinite(loss_train):
                checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch' : epoch,
                    }
                if args.checkpointing_path:
                    try:
                        torch.save(checkpoint, args.checkpointing_path + identifier_ + '_model.pt')
                    except Exception as e: 
                        print(e)
                        writer.add_text("Exceptions", str(e))
            else:
                print('oh no, loss was NaN. Reloading last epoch version')

            writer.add_scalar(f'GradientNormIsometryRegularization', isometry_regularization_grad_norm, epoch)
            writer.add_scalar(f'GradientNormTotal', average_gradient_last_epoch, epoch)
            writer.add_scalar(f'GradientNonfiniteFraction', non_finite_gradient_fraction, epoch)

            writer.add_scalar(f'Loss/train', loss_train, epoch)
            writer.add_scalar(f'Accuracy/train', accuracy_train, epoch)
            writer.flush()

        if checkpoint['epoch'] <= epoch: # recomputes last epoch's metrics after interruption to add hyperparameters if failed at that stage
            # validate on maximum superposition capacity
            model.num_img_sup = args.num
            loss_validation_at_capacity, accuracy_validation_at_capacity = validate_epoch()

            # validate on sup_low images superposed
            if sup_low != args.num:
                model.num_img_sup = sup_low
                loss_validation_at_sup_low, accuracy_validation_at_sup_low = validate_epoch()
            else:
                loss_validation_at_sup_low = 0
                accuracy_validation_at_sup_low = 0

            # model parameters
            absa = model.compute_average_abs_alpha(device) if isinstance(model, SuperWideISONet) else 0
            rpa = model.compute_average_relu_param(device) if isinstance(model, SuperWideISONet) else 0
            rpv = model.compute_relu_param_variance(device) if isinstance(model, SuperWideISONet) else 0

            # training setting and regularization values
            br = model.binding_regularization()
            ir = model.isometry_regularization(device)

        if checkpoint['epoch'] <= epoch:
            writer.add_scalar(f'Loss/validation_at_capacity', loss_validation_at_capacity, epoch)
            writer.add_scalar(f'Accuracy/validation_at_capacity', accuracy_validation_at_capacity, epoch)
            writer.add_scalar(f'Loss/validation_at_sup_low', loss_validation_at_sup_low, epoch)
            writer.add_scalar(f'Accuracy/validation_at_sup_low', accuracy_validation_at_sup_low, epoch)
            writer.add_scalar(f'AbsoluteAlpha', absa, epoch)
            writer.add_scalar(f'AverageReLUParam', rpa, epoch)
            writer.add_scalar(f'VarianceReLUParam', rpv, epoch)
            writer.add_scalar(f'LearningRate', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar(f'Binding_Regularization', br, epoch)
            writer.add_scalar(f'Isometry_Regularization', ir, epoch)
            writer.flush()


    # in case of dynamic use case log metrics on fewer images.
    metric_dict = {
        'accuracy validation at capacity' : accuracy_validation_at_capacity, 
        'loss validation at capacity' : loss_validation_at_capacity,
        'accuracy validation at sup_low' : accuracy_validation_at_sup_low, 
        'loss validation at sup_low' : loss_validation_at_sup_low,
        'average absolute alpha' : absa,
        'average ReLU parameter': rpa,
        'variance ReLU parameter': rpv,
        'binding regularization' : br,
        'isometry regularization' : ir,

    }
    writer.add_hparams(description_dict, metric_dict)
    writer.close()

