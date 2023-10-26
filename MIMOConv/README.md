# MIMOConv

This folder contains the code used to reproduce the MIMOConv results. 

### Hardware
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels. We tested our methods on an NVIDA Tesla A100 GPU with CUDA Version 11.3.1. 

### Installing Dependencies

The `conda` software is required for running the code. Generate a new environment with

```
$ conda create --name myMIMOenv python=3.7
$ conda activate myMIMOenv
```

We need PyTorch 1.11 and CUDA. 

```
$ (myMIMOenv) conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge
$ (myMIMOenv) pip install -r requirements.txt
```

## CIFAR10/100 dataset
The dataset should be downloaded automatically at first usage of the script. If it does not, you can download the CIFAR10 and CIFAR100 dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html). Store the files under `/data`. 

## CNNs
The main file to run experiments is `src/train_superposition.py`. It has the following help page:

```
    usage: train_superposition.py [-h] -x EXPERIMENT_NR [-f] [-w WIDTH] [-i INITIAL_WIDTH] [-p RELU_PARAMETER_INIT] [-k] [-q] [-z] [-v] [-j BINDING_REGULARIZATION] [-o ORTHOGONAL_REGULARIZATION]
                                  [-u SUP_FREQUENCY] [-s SUP_LOW] [-b BATCH_SIZE] [-n NUMBER_OF_CPUS] [-e EPOCHS] [-l LR] [-d WEIGHT_DECAY] [-a MIXUP_ALPHA] [-g GRADIENT_CLIPPING] [-c CHECKPOINTING_PATH]
                                  [-t TRANSFER_LEARNING_PATH] [-r RANDOM_SEED]
                                  {WideResNet-28,WideISOReLUNet-28,WideISONet-28,MIMONet-28,WideResNet-16,WideISOReLUNet-16,WideISONet-16,MIMONet-16} {CIFAR10,CIFAR100} {None,HRR,MBAT} num

    Trains and evaluates CNNs, in particular demonstrating superposition principles

    positional arguments:
      {WideResNet-28,WideISOReLUNet-28,WideISONet-28,MIMONet-28,WideResNet-16,WideISOReLUNet-16,WideISONet-16,MIMONet-16}
                            architecture and network depth
      {CIFAR10,CIFAR100}    dataset
      {None,HRR,MBAT}       binding type
      num                   maximum superposition capability of model

    optional arguments:
      -h, --help            show this help message and exit
      -x EXPERIMENT_NR, --experiment_nr EXPERIMENT_NR
                            useful for storing results in separate folders (default: None)
      -f, --final           run on train/test split instead of train/val split - should only be enabled at the very end of hypertuning (default: False)
      -w WIDTH, --width WIDTH
                            width of network, i.e. factor to increase channel size (default: 1)
      -i INITIAL_WIDTH, --initial_width INITIAL_WIDTH
                            width of output of initial convolutional layer, which is not affected by [width] (default: 1)
      -p RELU_PARAMETER_INIT, --relu_parameter_init RELU_PARAMETER_INIT
                            offset in shiftedReLU (default -1) or parameter in parametricReLU (default 0.5) depending on the model (default: None)
      -k, --skip_init       whether skip_init (ignore residual branch at initialisation) is enabled (default: False)
      -q, --dirac_init      whether dirac initialisation of convolutions is enabled (default: False)
      -z, --batch_norm_disabled
                            whether batch norm is disabled (default: False)
      -v, --trainable_keys_disabled
                            whether keys are fixed after initialisation (default: False)
      -j BINDING_REGULARIZATION, --binding_regularization BINDING_REGULARIZATION
                            regularization factor to produce orthogonal unit norm HRR vectors (default: 0.1)
      -o ORTHOGONAL_REGULARIZATION, --orthogonal_regularization ORTHOGONAL_REGULARIZATION
                            regularization factor keeping convolutions isometric in MIMONet and WideISONet (default: 0.0001)
      -u SUP_FREQUENCY, --sup_frequency SUP_FREQUENCY
                            fraction of training batches with [num] images are superposed (instead of [sup_low]) should be between 0 and 1 (default: 1.0)
      -s SUP_LOW, --sup_low SUP_LOW
                            how many images are superposed in the low-demand setting. If left unspecified always [num] images are superposed (default: None)
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            the batch size used. The batch size before binding is larger by a factor [num] (default: 128)
      -n NUMBER_OF_CPUS, --number_of_cpus NUMBER_OF_CPUS
                            number of cpus used in dataloading (default: 8)
      -e EPOCHS, --epochs EPOCHS
                            the number of epochs during training. Note that for a fair comparison each epoch passes through data [num] times (default: 200)
      -l LR, --lr LR        the maximal learning rate (oneCycleLR Policy) (default: 0.2)
      -d WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                            weight_decay, i.e. l2 regularization. Not applied to relu parameters (default: 1e-05)
      -a MIXUP_ALPHA, --mixup_alpha MIXUP_ALPHA
                            high alpha yields strong mixup, alpha = 1 uniform mixup, alpha = 0 no mixup (default: 1.0)
      -g GRADIENT_CLIPPING, --gradient_clipping GRADIENT_CLIPPING
                            the cutoff ratio (compared to average gradient norm of last epoch) selecting which batches are filtered out (default: 10.0)
      -c CHECKPOINTING_PATH, --checkpointing_path CHECKPOINTING_PATH
                            saves model to path after each epoch (default: /dccstor/saentis/MIMONet/)
      -t TRANSFER_LEARNING_PATH, --transfer_learning_path TRANSFER_LEARNING_PATH
                            loads model from checkpoint at path as starting point (default: None)
      -r RANDOM_SEED, --random_seed RANDOM_SEED
                            allows reproducibility (default: 42)
```

Below, you can find the commands to generate the main results (see Table 1 in paper): 

```
# WideResNet-28-10 (training for 200 epochs yielded better results for this model) 
$(myMIMOenv) python src/train_superposition.py -f -e 200 -x 0 -r 1 --width 10 --initial_width 1 WideResNet-28 CIFAR10 None 1
$(myMIMOenv) python src/train_superposition.py -f -e 200 -x 0 -r 1 --width 10 --initial_width 1 WideResNet-28 CIFAR100 None 1

# WideIsoNet-28-10 
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -x 1 -r $RUN --width 10 --initial_width 1 WideISOReLUNet-28 CIFAR10 None 1
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -x 1 -r $RUN --width 10 --initial_width 1 WideISOReLUNet-28 CIFAR100 None 1

# MIMOConv static N=1
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -f -x 2 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR10 HRR 1
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -f -x 2 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR100 HRR 1

# MIMOConv static N=2
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -f -x 3 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR10 HRR 2
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -f -x 3 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR100 HRR 2

# MIMOConv static N=4
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -f -x 4 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR10 HRR 4
$(myMIMOenv) python src/train_superposition.py -f -e 1200 -f -x 4 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR100 HRR 4

# MIMOConv dynamic N1--4 (for inference see later)
$(myMIMOenv) python src/train_superposition.py --sup_low 1 --sup_frequency 0.8 -e 1200 -f -x 5 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR10 HRR 4
$(myMIMOenv) python src/train_superposition.py --sup_low 1 --sup_frequency 0.8 -e 1200 -f -x 5 -r 1 --width 10 --initial_width 4 MIMONet-28 CIFAR100 HRR 4
```


The results are written to a tensorboard session under `runs/`. The file `src/train_superposition.py` uses several helperfiles. `src/datasets.py` is used to preprocess and load the datasets CIFAR10/100. `src/mixup.py` implements [Mixup Data Augmentation](https://arxiv.org/abs/1710.09412). In `src/optimizer.py` the optimizer is configured and `src/superposition.py` acts as a parent class to all superposition enabled CNNs, implementing binding and bundling. Finally, in the folder models the file `superwideresnet.py` implements a superposition capable WideResNet and `superwideisonet.py` inherits from it and adds isometry-preserving activation functions and convolutions.

The file `src/validate_dynamic.py` is similar to `src/train_superposition.py` in spirit, however only used to evaluate models already trained. In particular, it allows one to evaluate dynamic models in modes that were never used during training. It has the following help page:

    usage: validate_dynamic.py [-h] [-w WIDTH] [-i INITIAL_WIDTH] [-p RELU_PARAMETER_INIT] [-k] [-q] [-z] [-v] [-s SUP_LOW] [-b BATCH_SIZE] [-n NUMBER_OF_CPUS] [-c CHECKPOINT]
                               {WideResNet-28,WideISOReLUNet-28,WideISONet-28,MIMONet-28,WideResNet-16,WideISOReLUNet-16,WideISONet-16,MIMONet-16} {CIFAR10,CIFAR100} {None,HRR,MBAT} num

    Evaluates CNNs, in particular demonstrating superposition principles

    positional arguments:
      {WideResNet-28,WideISOReLUNet-28,WideISONet-28,MIMONet-28,WideResNet-16,WideISOReLUNet-16,WideISONet-16,MIMONet-16}
                            architecture and network depth
      {CIFAR10,CIFAR100}    dataset
      {None,HRR,MBAT}       binding type
      num                   maximum superposition capability of model

    optional arguments:
      -h, --help            show this help message and exit
      -w WIDTH, --width WIDTH
                            width of network, i.e. factor to increase channel size (default: 1)
      -i INITIAL_WIDTH, --initial_width INITIAL_WIDTH
                            width of output of initial convolutional layer, which is not affected by [width] (default: 1)
      -p RELU_PARAMETER_INIT, --relu_parameter_init RELU_PARAMETER_INIT
                            offset in shiftedReLU (default -1) or parameter in parametricReLU (default 0.5) depending on the model (default: None)
      -k, --skip_init       whether skip_init (ignore residual branch at initialisation) is enabled (default: False)
      -q, --dirac_init      whether dirac initialisation of convolutions is enabled (default: False)
      -z, --batch_norm_disabled
                            whether batch norm is disabled (default: False)
      -v, --trainable_keys_disabled
                            whether keys are fixed after initialisation (default: False)
      -s SUP_LOW, --sup_low SUP_LOW
                            how many images are superposed in the low-demand setting. If left unspecified always [num] images are superposed (default: None)
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            the batch size used. The batch size before binding is larger by a factor [num] (default: 128)
      -n NUMBER_OF_CPUS, --number_of_cpus NUMBER_OF_CPUS
                            number of cpus used in dataloading (default: 8)
      -c CHECKPOINT, --checkpoint CHECKPOINT
                            which model weights to load (default: )

It is important to give the script the same model hyperparameters (such as width etc.) as the model which is loaded from a checkpoint. 

For validating the dynamic model, run the following commands
```
# slow
$(myMIMOenv) python src/validate_dynamic.py --sup_low 1 -c "path/to/model/CIFAR10model.pt" --width 10 --initial_width 4 MIMONet-28 CIFAR10 HRR 4
$(myMIMOenv) python src/validate_dynamic.py --sup_low 1 -c "path/to/model/CIFAR100model.pt" --width 10 --initial_width 4 MIMONet-28 CIFAR100 HRR 4

# normal
$(myMIMOenv) python src/validate_dynamic.py --sup_low 2 -c "path/to/model/CIFAR10model.pt" --width 10 --initial_width 4 MIMONet-28 CIFAR10 HRR 4
$(myMIMOenv) python src/validate_dynamic.py --sup_low 2 -c "path/to/model/CIFAR100model.pt" --width 10 --initial_width 4 MIMONet-28 CIFAR100 HRR 4

# fast
$(myMIMOenv) python src/validate_dynamic.py --sup_low 4 -c "path/to/model/CIFAR10model.pt" --width 10 --initial_width 4 MIMONet-28 CIFAR10 HRR 4
$(myMIMOenv) python src/validate_dynamic.py --sup_low 4 -c "path/to/model/CIFAR100model.pt" --width 10 --initial_width 4 MIMONet-28 CIFAR100 HRR 4
```