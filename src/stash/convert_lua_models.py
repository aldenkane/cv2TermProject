import torch
from torch.utils.serialization import load_lua
from torchvision import datasets, transforms

# Load Module from Lua (NOTE: Only supported by torch 0.4.1), Save as PyTorch Model
model = load_lua('../models/CNN3_p8_n8_split4_073000.t7')

dataset = datasets.ImageFolder('path', transform=transform)

transform = transforms.Compose([
                                transforms.Resize(255)
                                transforms.ToTensor
                                ])