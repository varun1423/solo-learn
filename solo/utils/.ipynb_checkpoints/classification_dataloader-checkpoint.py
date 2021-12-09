# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import pandas as pd
from PIL import Image

import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
import h5py
import numpy as np


def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }
    return pipeline


def prepare_transforms(dataset: str) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }

    tbc_pipeline = {
        "T_train": transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(size=256, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485), std=(0.228)),
            ]
        ),
        "T_val": transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }

    bach_pipeline = {
        "T_train": transforms.Compose(
            [transforms.RandomResizedCrop(size=400, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [transforms.Resize(400),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "tbc": imagenet_pipeline,
        "bach": bach_pipeline,
        "custom": custom_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        data_dir Optional[Union[str, Path]]: path where to download/locate the dataset.
        train_dir Optional[Union[str, Path]]: subpath where the training data is located.
        val_dir Optional[Union[str, Path]]: subpath where the validation data is located.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if data_dir is None:
        sandbox_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_dir / "datasets"
    else:
        data_dir = Path(data_dir)

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)

    if val_dir is None:
        val_dir = Path(f"{dataset}/val")
    else:
        val_dir = Path(val_dir)

    assert dataset in ["cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "tbc", "bach", "custom"]

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            data_dir / train_dir,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            data_dir / val_dir,
            train=False,
            download=download,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            data_dir / train_dir,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            data_dir / val_dir,
            split="test",
            download=download,
            transform=T_val,
        )

    elif dataset in ["imagenet", "imagenet100", "custom"]:
        train_dir = data_dir / train_dir
        val_dir = data_dir / val_dir

        train_dataset = ImageFolder(train_dir, T_train)
        val_dataset = ImageFolder(val_dir, T_val)

    elif dataset == "tbc":
        train_dir = train_dir
        val_dir = val_dir

        train_dataset = ThermalBarrierCoating_h5(train_dir, T_train)
        val_dataset = ThermalBarrierCoating_h5_val(val_dir, T_val)
        
    elif dataset == "bach":
        train_dir = train_dir
        val_dir = val_dir

        train_dataset = Bach_train(train_dir, T_train)
        val_dataset = Bach_val(val_dir, T_val)

    return train_dataset, val_dataset


class ThermalBarrierCoating_h5(Dataset):
    def __init__(self, train_dir, transform=None):
        self.h5 = h5py.File(train_dir / Path('final_split.h5'), 'r')
        self.transform = transform

    def __getitem__(self, index):
        image = Image.fromarray(self.h5['data_train'][index])
        y_label =self.h5['train_label'][index]
        if self.transform is not None:
            image = self.transform(image)
        return image, np.int64(y_label)

    def __len__(self):
        return self.h5['data_train'].shape[0]


class ThermalBarrierCoating_h5_val(Dataset):
    def __init__(self, val_dir, transform=None):
        self.h5 = h5py.File(val_dir / Path('final_split.h5'), 'r')
        self.transform = transform

    def __getitem__(self, index):
        image = Image.fromarray(self.h5['data_val'][index])
        y_label =self.h5['val_label'][index]
        if self.transform is not None:
            image = self.transform(image)
        return image, np.int64(y_label)

    def __len__(self):
        return self.h5['data_val'].shape[0]

class Bach_train(Dataset):
    def __init__(self, train_dir, transform=None): 
        self.csv_path = '/p/project/atmlaml/project_SSL_varun/bach/'
        self.csv = pd.read_csv(self.csv_path / Path('train_csv.csv'))
        self.label = self.csv['Id']
        self.image_ID = self.csv['Image_Name']
        self.transform = transform
        self.train_dir = train_dir

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.train_dir, self.image_ID[index]))
        y_label = self.label[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, y_label

    def __len__(self):
        return self.image_ID.shape[0]


class Bach_val(Dataset):
    def __init__(self, val_dir, transform=None):
        self.csv_path = '/p/project/atmlaml/project_SSL_varun/bach/'
        self.csv = pd.read_csv(self.csv_path / Path('csv_val.csv'))
        self.label = self.csv['Id']
        self.image_ID = self.csv['Image_Name']
        self.transform = transform
        self.val_dir = val_dir

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.val_dir, self.image_ID[index]))
        y_label = self.label[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, y_label

    def __len__(self):
        return self.image_ID.shape[0]
    

"""
class ThermalBarrierCoating_val(Dataset):
    def __init__(self, train_dir, transform=None):
        self.train_dir = train_dir
        self.data_csv = pd.read_csv(train_dir / Path('meta_data.csv')) 
        self.data_csv = self.data_csv.sample(n=1000).reset_index(drop=True)
        self.label = self.data_csv['sample_name']
        self.image_ID = self.data_csv['Image_Name']
        self.transform = transform

    def __getitem__(self, index):
        y_label = self.label[index]
        image = Image.open(os.path.join(self.train_dir, 'data_all_csv', self.image_ID[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, y_label

    def __len__(self):
        return self.data_csv['sample_name'].shape[0]

    

class ThermalBarrierCoating(Dataset):
    def __init__(self, train_dir, transform=None):
        self.train_dir = train_dir
        self.data_csv = pd.read_csv(train_dir / Path('meta_data.csv'))
        self.label = self.data_csv['sample_name']
        self.image_ID = self.data_csv['Image_Name']
        self.transform = transform

    def __getitem__(self, index):
        y_label = self.label[index]
        image = Image.open(os.path.join(self.train_dir, 'data_all_csv', self.image_ID[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, y_label

    def __len__(self):
        return self.data_csv['sample_name'].shape[0]
"""

def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        data_dir (Optional[Union[str, Path]], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): subpath where the
            training data is located. Defaults to None.
        val_dir (Optional[Union[str, Path]], optional): subpath where the
            validation data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """

    T_train, T_val = prepare_transforms(dataset)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        download=download,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
