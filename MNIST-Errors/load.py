# Lädt Davids MnistNetz, dass nur Gewichte hat, und speichert mit Struktur wieder ab

import torch.nn as nn
import torch

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# Load the state dict
state_dict = torch.load("MnistNetz.pth", weights_only=True)

# Create new state dict with modified keys
new_state_dict = {}
for key in state_dict:
    new_key = key.replace("model.", "")
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)
torch.save(model, "Davids_MNIST_Model.pt")
