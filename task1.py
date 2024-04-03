# Import necessary libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
# FashionMNIST is a dataset of Zalando's article images, with 60k training examples and 10k test examples.
# Each example is a 28x28 grayscale image, associated with a label from 10 classes.
training_data = datasets.FashionMNIST(
    root="data",  # Specifies the root directory of the dataset
    train=True,  # Specifies this data will be used for training the model
    download=True,  # Downloads the data from the internet if it's not available at root.
    transform=ToTensor(),  # Converts a PIL Image or numpy.ndarray to tensor.
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,  # Specifies this data will be used for testing the model
    download=True,
    transform=ToTensor(),
)

# Define the batch size for the data loader.
batch_size = 64

# Create data loaders.
# Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Print the shape of the data
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"  # CUDA is an API that allows for using the compute power of Nvidia GPUs
    if torch.cuda.is_available()
    else "mps"  # MPS is a feature to allow multiple CUDA processes to share a single GPU context
    if torch.backends.mps.is_available()
    else "cpu"  # If neither CUDA nor MPS is available, use CPU
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # Flattens the input. Does not affect the batch size.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # Applies a linear transformation to the incoming data
            nn.ReLU(),  # Applies the rectified linear unit function element-wise
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize the NeuralNetwork and move the model to the device
model = NeuralNetwork().to(device)
print(model)

# Define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class.
optimizer = torch.optim.SGD(model.parameters(), lr=1)  # Stochastic Gradient Descent

# Define the training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # Set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # Resets the gradients to zero
        loss.backward()  # Computes the gradient of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        optimizer.step()  # Performs a single optimization step (parameter update)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define the testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Set the model to evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad():  # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # Pass the data through the model to get predictions
            test_loss += loss_fn(pred, y).item()  # Compute the loss and add it to the total test loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Count the number of correct predictions
    test_loss /= num_batches  # Compute the average test loss
    correct /= size  # Compute the accuracy
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # Print the test error, accuracy, and average loss