import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_sample = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_sample

dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data

dataloader = DataLoader(dataset=dataset, 
                        batch_size=4, shuffle=True)

# training loop

n_epochs = 2
total_samples = len(dataset)
n_itarations = math.ceil(total_samples / 4)

for epoch in range(n_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, step {i+1}/{n_itarations}, inputs {inputs.shape}")        

