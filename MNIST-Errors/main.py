import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import datasets, transforms
import numpy as np
import math
import matplotlib as plt

ACTIVATION_FN = nn.Tanh


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    x = torch.ones(1, device=device)

    return device


def get_data() -> datasets.MNIST:
    transform = transforms.Compose(
        [
            # Converts to float and normalizes from [0, 255] to [0, 1]
            transforms.ToTensor(),
            # Flattens the 2D image 28x28 to 1D vector 784
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_set = datasets.MNIST("data/", download=True, train=True, transform=transform)
    return train_set


def accuracy(x: torch.Tensor, y: torch.Tensor, model: nn.Module):
    model.eval()
    with torch.no_grad():
        prediction = model(x)

    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y

    return is_correct.cpu().numpy().tolist()


def main():
    device = get_device()
    model = torch.load("MNIST-Errors/MNIST_model.pt", weights_only=False)
    model.eval()
    train_data = get_data()
    train_data_size = train_data.data.shape[0]
    error_sums_per_class = np.zeros((10, 28, 28))
    class_counters = np.zeros(10)

    for index, sample in enumerate(train_data):
        image = sample[0].to(device)
        label = sample[1]
        class_counters[label] += 1
        is_correct = accuracy(image, label, model)
        if is_correct:
            for i, pixel_val in enumerate(image):
                faulted_copy = image.detach().clone()
                faulted_copy[i] = 1 - pixel_val
                is_correct = accuracy(faulted_copy, label, model)
                error_sums_per_class[label, math.floor(i / 28), i % 28] += 1
        print(f"Progress: {index}/{train_data_size} ({index / train_data_size * 100}%)");

    for i in range(10):
        for j in range(28):
            for k in range(28):
                error_sums_per_class[i][j][k] /= class_counters[i]
        plt.imshow(error_sums_per_class, interpolation="nearest")
    plt.show()


if __name__ == "__main__":
    main()
