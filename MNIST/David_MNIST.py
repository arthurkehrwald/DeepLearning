import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os

epochs = []
train_errors = []
test_errors = []

# Interaktiver Plot
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Schritte')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Train/Test-Accuracy während des Trainings')

train_line, = ax.plot([], [], 'bo-', label='Train Accuracy')
test_line, = ax.plot([], [], 'ro-', label='Test Accuracy')
ax.legend()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

def track():
    train_line.set_data(range(len(train_errors)), train_errors)
    test_line.set_data(range(len(test_errors)), test_errors)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def getPercentage(output, target):
    _, prediction = torch.max(output, 1)
    correct = torch.sum(prediction == target).item()
    full = target.size(0)
    return (correct / full) * 100

def testMicro(net, loader):
    net.eval()
    with torch.no_grad():
        data, target = next(iter(loader))
        output = net(data)
        _, prediction = torch.max(output, 1)
        correct = torch.sum(prediction == target).item()
        full = target.size(0)
        return 100 * (correct / full)

def test(net, loader):
    net.eval()
    correct = 0
    full = 0
    with torch.no_grad():
        for data, target in loader:
            output = net(data)
            _, prediction = torch.max(output, 1)
            correct += torch.sum(prediction == target).item()
            full += target.size(0)
    return 100 * (correct / full)


def train(net, train_loader, test_loader, epochs_total=5):
    net.train()
    step = 0
    for epoch in range(epochs_total):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc = getPercentage(output, target)
            # Tracking
            #train_errors.append(acc)
            #test_errors.append(testMicro(net,test_loader))
            # print(f"Epoch {epoch+1} Batch {batch_idx+1}: Train Acc = {acc:.2f}%")
            #track()
            optimizer.step()

        for param_group in optimizer.param_groups:
            print("Aktuelle LR:", param_group['lr'])

        print(f"Epoch {epoch+1}: Train Acc = {testMicro(net,test_loader):.2f}%")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)


    train(net, train_loader, test_loader, epochs_total=3)

    project_dir = os.path.abspath(os.getcwd())
    filepath = os.path.join(project_dir, "MnistNetz.pth")

    torch.save(net.state_dict(), filepath)

