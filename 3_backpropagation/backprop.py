import torch

# Chain rule
# x = a(x) -> 1 -> b(1) -> z =compute derivative 
# Computational Graph

# 1. Forward pass: Compute Loss
# 2. Compute local gradients
# 3. Backward pass: Compute dLoss / dWeights using the chain rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward pass and compute loss
y_hat = w * x
loss = ((y_hat - y) ** 2)
print(loss)

# Backward pass
loss.backward()
print(w.grad)


