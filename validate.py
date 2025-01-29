import torch
import config

def validate(model, validloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_loss / len(validloader)
    val_acc = 100 * correct_val / total_val

    return val_loss, val_acc