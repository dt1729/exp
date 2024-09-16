import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import Tensor

class NewFilter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(NewFilter, self).__init__()
        kernel_tensor = 0.2 * torch.ones(
            [out_channels, in_channels, kernel_size, kernel_size], dtype=torch.float32
        )
        self.kernel = nn.Parameter(kernel_tensor, requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, self.kernel, padding=1)
        return x

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.filt = NewFilter(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        self.feedback = None  # Initialize feedback as None

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [batch_size, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [batch_size, 64, 7, 7]

        # Initialize self.feedback if it's None or has different shape
        if self.feedback is None or self.feedback.shape != x.shape:
            self.feedback = torch.zeros_like(x)

        # Subtract self.feedback from x
        x = self.pool(F.relu(self.filt(x - self.feedback)))  # Shape: [batch_size, 64, 3, 3]

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Update self.feedback
        self.feedback = x.detach()

        return x

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the training data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(
    root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False)

# Initialize the network, define the criterion and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and targets to the specified device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)   # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()        # Backward pass
        optimizer.step()       # Update weights

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\
 Loss: {loss.item():.6f}')

# Testing loop
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # No need to compute gradients during testing
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Sum up batch loss
            test_loss += criterion(output, target).item()
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # Count correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: \
{correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

# Run training and testing
for epoch in range(1, 6):  # Train for 5 epochs
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    torch.save(model, 'trained_model.pth')

