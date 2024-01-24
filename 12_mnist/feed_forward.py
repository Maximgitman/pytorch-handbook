from pickletools import optimize
from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parametres
input_size = 784 #28x28
hidden_size = 100
n_classes = 10
n_epochs = 10
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), 
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                           transform=transforms.ToTensor(), 
                                           download=True)

from torch.utils.data import Dataset, DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

example = iter(train_loader)
sample, labels = example.next()
print(sample.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(sample[i][0], cmap="gray")
# plt.show()

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NN(input_size, hidden_size, n_classes)
critireon = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(n_epochs):
    model.train()
    for i, (img, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # input is 100, 784
        img = img.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(img)
        loss = critireon(outputs, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch {epoch +1} / {n_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")

# test

with torch.no_grad():
    model.eval()
    n_correct = 0
    n_sample = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, pred = torch.max(outputs, 1)
        n_sample += labels.shape[0]
        n_correct += (pred == labels).sum().item()

    acc = 100.0 * n_correct / n_sample
    print(f"accuracy {acc}")