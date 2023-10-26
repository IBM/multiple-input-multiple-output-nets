#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torchvision
from torch.utils import data
from torch import Generator
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
class DataSetWithTransforms(data.Dataset):
    r"""Wrapper Class to apply transforms (needed because data.random_split returns not separate datasets, but rather subsets sharing transforms)"""
    def __init__(self, dataset, transform=None, target_transform=None):
        r""" Initialisation
        Args:
        dataset: dataset to hold
        transform: transforms to apply to data
        target_transform: transforms to apply to target
        """
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        r"""Returns one item from dataset
        Args:
            idx: id of item
        Returns:
            item from dataset at position idx
        """
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        r"""Returns length of dataset
        Returns:
            length of dataset
        """
        return len(self.dataset)

def get_train_eval_test_sets(dataset:str, final_run:bool):
    r"""Data standardisation and data augmentation for the different datasets
    Args:
        dataset: either CIFAR10 or CIFAR100
        final_run: decides if random train/validation split is used or train/test split
    Returns:
        tuple of train and validation/test dataset of type DataSetWithTransforms
    """
    if dataset == "CIFAR10":
        trainevalset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        if final_run:
            trainset = trainevalset
            evalset = testset
        else:
            trainset, evalset = data.random_split(trainevalset, [45000, 5000], generator=Generator().manual_seed(42))
        data_channel_means = [0.4914, 0.4822, 0.4465]
        data_channel_standard_deviations = [0.2470, 0.2435, 0.2616]
        data_augmentation = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4, padding_mode="reflect")])
        transform_to_normal_tensor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=data_channel_means,
                             std=data_channel_standard_deviations, inplace=True)])
        augment_and_transform_to_normal_tensor = torchvision.transforms.Compose(
        [data_augmentation,
        transform_to_normal_tensor])
    elif dataset == "CIFAR100":
        trainevalset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
        if final_run:
            trainset = trainevalset
            evalset = testset
        else:
            trainset, evalset = data.random_split(trainevalset, [45000, 5000], generator=Generator().manual_seed(42))
        data_channel_means = [0.5071, 0.4867, 0.4408]
        data_channel_standard_deviations = [0.2675, 0.2565, 0.2761]
        data_augmentation = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4, padding_mode="reflect")])
        transform_to_normal_tensor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=data_channel_means,
                             std=data_channel_standard_deviations, inplace=True)])
        augment_and_transform_to_normal_tensor = torchvision.transforms.Compose(
            [data_augmentation,
            transform_to_normal_tensor])
    elif dataset == "MNIST": 
        trainevalset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        if final_run:
            trainset = trainevalset
            evalset = testset
        else:
            trainset, evalset = data.random_split(trainevalset, [55000, 5000], generator=Generator().manual_seed(42))
        data_channel_means = [0.1307]
        data_channel_standard_deviations = [0.3081]
        data_augmentation = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()])
        transform_to_normal_tensor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(20),
            torchvision.transforms.Normalize(mean=data_channel_means,
                             std=data_channel_standard_deviations, inplace=True)])
        augment_and_transform_to_normal_tensor = torchvision.transforms.Compose(
            [data_augmentation,
            transform_to_normal_tensor])
    elif dataset == "SVHN": 
        data_size = 32
        trainevalset = torchvision.datasets.SVHN(root='/dccstor/saentis/data/', split="train", download=True)
        testset = torchvision.datasets.SVHN(root='/dccstor/saentis/data/', split="test", download=True)
        if final_run:
            trainset = trainevalset
            evalset = testset
        else:
            trainset, evalset = data.random_split(trainevalset, [65000, 8257], generator=Generator().manual_seed(42))

        normalize = transforms.Normalize(
                    mean = [x / 255 for x in [109.9, 109.7, 113.8]],
                    std = [x / 255 for x in [50.1, 50.6, 50.8]]) 

        augment_and_transform_to_normal_tensor = transforms.Compose([
                        transforms.RandomCrop((data_size, data_size), padding=2),
                        transforms.ToTensor(),
                        normalize])
        transform_to_normal_tensor = transforms.Compose([
                        transforms.ToTensor(),
                        normalize])
    else:
        raise Exception(f"Unknown Dataset {dataset}")    

    trainset = DataSetWithTransforms(trainset, transform=augment_and_transform_to_normal_tensor)
    evalset = DataSetWithTransforms(evalset, transform=transform_to_normal_tensor)
    return trainset, evalset