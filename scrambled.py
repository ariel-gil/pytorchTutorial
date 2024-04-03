import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lrs = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.lrs(x)
        return logits

def train(dataloader, m, f, o, s):
    size = len(dataloader.dataset)
    m.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        p = m(x)
        loss = f(p, y)
        loss.backward()
        o.step()
        o.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    s.step()

def test(dataloader, m, f):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    m.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            p = m(x)
            test_loss += f(p, y).item()
            correct += (p.argmax(1) == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 15
lr = 0.001
m = NN().to(device)
f = nn.CrossEntropyLoss()
o = torch.optim.Adam(m.parameters(), lr=lr)
s = torch.optim.lr_scheduler.StepLR(o, step_size=5, gamma=0.1)

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

dl1 = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
dl2 = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=64, shuffle=True)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dl1, m, f, o, s)
    test(dl2, m, f)

print("Done!")