from cProfile import label
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.n_sample = xy.shape[0]

        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.tramsform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        
        if self.tramsform:
            sample = self.tramsform(sample)
        return sample

    def __len__(self):
        return self.n_sample


class ToTensor():

    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)

class MulTransform():
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)