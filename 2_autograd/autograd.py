import torch

x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y*2
# z = z.mean()
print(z)

v = torch.tensor([0.1, 1.9, 0.001], dtype=torch.float32)
z.backward(v) # dz/dx
print(x.grad)

# Privent from tracking gradient
# 1. x.requers_grad(False)
# 2. x.detach()
# 3. with torch.no_grad():

x = torch.randn(3, requires_grad=False)
x.requires_grad_(False)
print(x)

y = x.detach()
print(y)

with torch.no_grad():
    y = x + 2
    print(y)