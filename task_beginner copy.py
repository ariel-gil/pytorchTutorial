import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

d1 = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
d2 = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

dl1 = DataLoader(d1, batch_size=64)
dl2 = DataLoader(d2, batch_size=64)



class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lrs = nn.Sequential(
            nn.Linear(784*1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.lrs(x)
        return logits

m = NN().cuda()

f = nn.CrossEntropyLoss()
o = torch.optim.SGD(m.parameters(), lr=0.1)

def train(dataloader, m, f, o):
    size = len(dataloader.dataset)
    m.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        p = m(x)
        loss = f(p, y)
        loss.backward()
        o.step()
        o_zero_grad = 0
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            o.zero_grad()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, m, f):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0
    test_loss = 0
    for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            p = m(x)
            test_loss += f(p, y).item()
            correct += (p.argmax(1) == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 1

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dl1, m, f, o)
    test(dl2, m, f)
print("Done!")
