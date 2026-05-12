import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class UrbanSoundDataset(Dataset):
    def __init__(self, csv_path, fold_list, root_dir):
        df = pd.read_excel(csv_path)
        temp_info = df[df['fold'].isin(fold_list)]
        self.root_dir = Path(root_dir)

        exists = temp_info.apply(lambda x: (self.root_dir / f"fold{int(x['fold'])}" / x['file_name']).exists(), axis=1)
        self.data_info = temp_info[exists].reset_index(drop=True)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        row = self.data_info.iloc[index]
        file_name = row['file_name']
        fold = row['fold']
        file_path = self.root_dir / f"fold{fold}" / file_name
        label = int(row['label']) - 1

        data = np.load(file_path)

        target_width = 126
        current_width = data.shape[1]

        if current_width < target_width:
            pad_width = target_width - current_width
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        elif current_width > target_width:
            data = data[:, :target_width]

        data = torch.from_numpy(data).float().unsqueeze(0)
        return data, torch.tensor(label).long()


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResSiren(nn.Module):
    def __init__(self):
        super(ResSiren, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        self.blk1 = ResBlk(32, 64, stride=1)
        self.blk2 = ResBlk(64, 128, stride=1)
        self.blk3 = ResBlk(128, 256, stride=1)
        self.blk4 = ResBlk(256, 512, stride=1)
        self.outlayer = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        return self.outlayer(x)


criterion = nn.CrossEntropyLoss()


def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_index % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_index * len(data)}/{len(train_loader.dataset)} ({100. * batch_index / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    return total_loss / len(train_loader)


def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}%\n")
    return accuracy


if __name__ == '__main__':
    CSV_PATH = r"C:\Users\Petra\Desktop\ujm\projects\Code\total_data.xlsx"
    ROOT_DIR = r"D:\Petra\UrbanSound8K\UrbanSound8K\total_mel_dataset"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 10

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fold_group = [folds[i:i + 2] for i in range(0, len(folds), 2)]

    print("Start training!")
    final_results = []

    for i, test_folds in enumerate(fold_group):
        print(f"--- Round {i + 1} / {len(fold_group)} (Test Folds: {test_folds}) ---")
        train_folds = [f for f in folds if f not in test_folds]

        train_dataset = UrbanSoundDataset(CSV_PATH, train_folds, ROOT_DIR)
        test_dataset = UrbanSoundDataset(CSV_PATH, test_folds, ROOT_DIR)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ResSiren().to(DEVICE)
        optimize = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        test_accuracies = []
        best_accuracy = 0

        for epoch in range(1, EPOCHS + 1):
            avg_loss = train_model(model, DEVICE, train_loader, optimize, epoch)
            accuracy = test_model(model, DEVICE, test_loader)

            train_losses.append(avg_loss)
            test_accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "ressiren_weights.pth")

        final_results.append(best_accuracy)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, EPOCHS + 1), train_losses, 'r-o', label='Train Loss')
        plt.title(f'Round {i + 1} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, EPOCHS + 1), test_accuracies, 'b-o', label='Test Acc')
        plt.title(f'Round {i + 1} - Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'round_{i + 1}.png')

    print("--- All Rounds Completed! ---")
    print(f"Best Accuracies per Round: {final_results}")
    print(f"Mean Accuracy: {np.mean(final_results):.2f}%")
