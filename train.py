import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Net
import matplotlib.pyplot as plt
import config
from dataset_loader import trainloader, validloader
from validate import validate
from visualize import visualize

def train(model, criterion, optimizer, scheduler=None):
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config.EPOCHS}], Batch [{batch_idx+1}/{len(trainloader)}], "
                      f"Loss: {loss.item():.4f}")

        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = validate(model, validloader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler:
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pth")
            print(f"New best model saved with Val Acc: {val_acc:.2f}%")

        print(f"Epoch [{epoch+1}/{config.EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    visualize(train_losses, val_losses, train_accs, val_accs)
