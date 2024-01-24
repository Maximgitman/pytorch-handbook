"""
Steps fot torch model
1. Design model (input, output size, forward pass)
2. Construct loss and optimizer
3. Training loop:
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
"""
# 1. Import modeules
import torch
import torch.nn as nn

# 2. Getting data
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# 3. Test data
X_test = torch.tensor([5], dtype=torch.float32)

# 4. Init params for model and data
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features
learning_rate = 0.01
n_iters = 100
print(n_samples, n_features)

# model = nn.Linear(input_size, output_size)

# 5. Design model 
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# 6. Loss function and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 7. Training loop
print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")
for epoch in range(n_iters):
    # Predict
    y_pred = model(X)

    # Compute loss
    l = loss(Y, y_pred)

    # backword pass
    l.backward()
    
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")