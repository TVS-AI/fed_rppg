from typing import List, Tuple, cast

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchrppg.dataset.dataset_loader import dataset_loader
from torch.utils.data import DataLoader, TensorDataset

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]

def ubfc_deepphy_to_numpy() -> Tuple[XY, XY]:
    xy_train_input = []
    xy_train_target = []
    trainSet = dataset_loader(option="train")
    for trainData in trainSet:
        xy_train_input.append(trainData[0].numpy())
        xy_train_target.append(trainData[1].numpy())
    xy_train = np.array(xy_train_input), np.array(xy_train_target).reshape(-1)

    xy_test_input = []
    xy_test_target = []
    testSet = dataset_loader(option="test")
    for testData in testSet:
        xy_test_input.append(testData[0].numpy())
        xy_test_target.append(testData[1].numpy())
    xy_test = np.array(xy_test_input), np.array(xy_test_target).reshape(-1)

    return xy_train, xy_test

def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split x and y into a number of partitions."""
    return list(
        zip(np.array_split(x, num_partitions), np.array_split(y, num_partitions))
    )


def create_partitions(source_dataset: XY, num_partitions: int) -> XYList:
    """Create partitioned version of a source dataset."""
    x, y = source_dataset
    x, y = shuffle(x, y)
    xy_partitions = partition(x, y, num_partitions)

    return xy_partitions


def load(num_partitions: int, batch_size: int, shuffle: bool) -> PartitionedDataset:
    """Create partitioned version of CIFAR-10."""
    xy_train, xy_test = ubfc_deepphy_to_numpy()
    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)
    list_of_dataloaders = []
    for xy_train, xy_test in zip(xy_train_partitions, xy_test_partitions):
        x_train, y_train = xy_train
        x_test, y_test = xy_test

        train_dl = DataLoader(
            TensorDataset(torch.Tensor(x_train), torch.FloatTensor(y_train)),
            batch_size=batch_size,
            shuffle=shuffle
        )
        test_dl = DataLoader(
            TensorDataset(torch.Tensor(x_test), torch.FloatTensor(y_test)),
            batch_size=batch_size,
            shuffle=shuffle
        )
        list_of_dataloaders.append((train_dl, test_dl))

    return list_of_dataloaders