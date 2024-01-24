from cProfile import label
from random import shuffle
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parametres
n_epochs = 10
batch_size = 16
learning_rate = 0.001

transform = transform.Compose(
    [transform.ToTensor(), 
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                    download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                    download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", 
            "frog", "horse", "ship", "truck")

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 3 images chanel, 
        self.conv1 = nn.Conv2d(3, 6, 5)
        # (32 - 5) + 0 /1 +1 6x28x28
        self.pool1 = nn.MaxPool2d(2, 2)
        # 6x14x14
        self.conv2 = nn.Conv2d(6, 16, 5)
        # (14 - 5) + 0 /1 +1 16x10x10
        self.pool2 = nn.MaxPool2d(2, 2)
        # 16x5x5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(n_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # oroginal shape [32, 3, 32, 32]
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")

print("Training finished")

with torch.no_grad():
    model.eval()
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        _, prediction = torch.max(output, 1)
        n_samples += labels.size(0)
        n_correct += (prediction == labels).sum().item()

        for i  in range(batch_size):
            label = labels[i]
            pred = prediction[i]
            if (label == pred):
                n_class_correct[label] +=1
            n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy is: {acc} %")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of class {classes[i]} is: {acc} %")
