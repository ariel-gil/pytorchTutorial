import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Private variable named the same as a main variable
        self._flatten = nn.Flatten()
        self.lrs = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.lrs(x)
        return logits

class train():
    def __init__(self, dataloader, m, f, o):
        size = len(dataloader.dataset)
        m.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            p = m(x)
            loss = f(p, y)
            loss.backward()
            o.step()
            o.zero_grad()

            # Overwriting global variable inside a function
            global global_var
            global_var = 20

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

class test():
    def __init__(self, dataloader, m, f):
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
        # Incorrect calculation of accuracy
        correct /= num_batches
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 1

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dl1, m, f, o)
    test(dl2, m, f)
print("Done!")