import torch
from torch.utils.data import random_split, Subset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import config

train_augs = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_augs = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(root=config.IMAGES_PATH, transform=None)

trainset, validset = random_split(dataset, [10000, 5000])

trainset.dataset.transform = train_augs
validset.dataset.transform = valid_augs

trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=config.BATCH_SIZE, shuffle=True)
