import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EMNIST
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

# ---------------------------
# Definicja modelu
# ---------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 27)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------------
# Funkcje pomocnicze
# ---------------------------
def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.0005, log_dir="logs"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        scheduler.step()
        val_acc = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch} | Avg Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}')

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    writer.close()
    return model


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


def save_checkpoint(model, optimizer, epoch, path='letter_model.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path='letter_model.pth'):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Model loaded from file, starting from epoch:", start_epoch)
        return model, optimizer, start_epoch
    else:
        print(f"No checkpoint found at {path}. Starting training from scratch.")
        return model, optimizer, 0


def load_model_for_inference(path='letter_model.pth'):
    """
    Load the model strictly for inference, extracting only the model's state_dict.
    """
    model = ConvNet()
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model weights loaded for inference.")
        else:
            raise RuntimeError("The checkpoint does not contain 'model_state_dict'. Make sure you provided the correct file.")
    else:
        print("No pre-trained model found. Please train the model first.")
    model.eval()
    return model



def get_data_loaders(batch_size=128, augment=True):
    transform_list = []
    if augment:
        transform_list += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ]
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    transform = transforms.Compose(transform_list)

    train_dataset = EMNIST('./data', split='letters', train=True, download=True, transform=transform)

    dataset_size = len(train_dataset)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

# ---------------------------
# Główna funkcja
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the ConvNet model.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoader.")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate.")
    parser.add_argument('--log_dir', type=str, default=f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Directory for TensorBoard logs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    need_training = not os.path.exists('letter_model.pth')

    model = ConvNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if need_training:
        train_loader, val_loader = get_data_loaders(batch_size=args.batch_size)
        trained_model = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, log_dir=args.log_dir)
        save_checkpoint(trained_model, optimizer, args.epochs - 1)
    else:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer)


if __name__ == "__main__":
    main()
