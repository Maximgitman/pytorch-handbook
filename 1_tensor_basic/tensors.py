import numpy as np
import torch 


# Creating an empty tensors
x = torch.empty(1)
z = torch.empty(2, 3)
y = torch.empty(2, 2, 4)

# Generate tensor with random float value
a = torch.rand(2, 3)
b = torch.zeros(2, 3)
c = torch.ones(2, 3)

# Change dtype for tensor
a = torch.rand(2, 3, dtype=torch.float16)
print(a.dtype)

# Prine size of tensot
print(a.size())

# Tensot from python list
g = torch.tensor([2.5, 0.2])
print(g)

# Simple addition tensor
x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x + y
z = torch.add(x, y) # Same as +
z = x - y
z = torch.sub(x, y) # Same as -
z = x * y
z = torch.mul(x, y) # Same as *
z = x / y
z = torch.div(x, y) # Same as /

# Modify tensor from another one
# All func in torch with underscore is inplace
y.add_(x) 
y.sub_(x) 
y.mul_(x) 
y.div_(x) 

# Sclicing tensor
x = torch.rand(5, 3)
print(x)
print(x[:, 0]) # all rows but first col
print(x[1, :]) # all cols but first row
print(x[1, 1]) # one element, on the 1st col and 1st row
print(x[1, 1].item()) # get actua value of tensor

# Reshape tensor
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)

# From numpy to torch and 
a = torch.ones(5)
print(a, type(a))
b = a.numpy()
print(b, type(b))

# If we use CPU then it will be tje same chank of memeory
a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    # tensor on GPU
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # You cannot to move GPU tensor back to numpy
    z = z.to("cpu")

x = torch.ones(5, requires_grad=True)
print(x)
