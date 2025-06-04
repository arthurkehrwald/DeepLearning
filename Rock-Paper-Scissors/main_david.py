import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

epochs = []
train_errors = []
test_errors = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def getPercentage(output, target):
    _, prediction = torch.max(output, 1)
    correct = torch.sum(prediction == target).item()
    return (correct / target.size(0)) * 100


def testMicro(net, loader):
    net.eval()
    with torch.no_grad():
        data, target = next(iter(loader))
        data, target = data.to(device), target.to(device)
        output = net(data)
        return getPercentage(output, target)


def test(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, prediction = torch.max(output, 1)
            correct += torch.sum(prediction == target).item()
            total += target.size(0)
    return 100 * correct / total


def train(net, train_loader, test_loader, epochs_total=5):
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs_total):
        net.train()
        print(f"\nEpoch {epoch+1}/{epochs_total}")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()

        val_acc = test(net, test_loader)
        test_errors.append(val_acc)

        print(f"Val Acc = {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(net.state_dict(), "best_model.pth")
            print("Bestes Modell gespeichert!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping ausgelöst!")
                break


data_dir = "Rock-Paper-Scissors/data"

transform_train = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ]
)
dataset = datasets.ImageFolder(root=data_dir)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=8, persistent_workers=True
)

val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True
)
print("Klassen:", dataset.classes)


if __name__ == "__main__":
    plt.clf()
    plt.cla()
    plt.close()

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train(net, train_loader, val_loader, epochs_total=100)
    print(test(net, val_loader))
    project_dir = os.path.abspath(os.getcwd())
    filepath = os.path.join(project_dir, "RockPaperNet.pth")

    torch.save(net, "netz_komplett.pth")
