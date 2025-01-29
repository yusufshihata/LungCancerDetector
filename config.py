import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Net

IMAGES_PATH = "/data"
BATCH_SIZE = 32
BETAS = (0.9, 0.999)
LR = 3e-4
EPOCHS = 30
DEVICE = "cuda"

model = Net().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
