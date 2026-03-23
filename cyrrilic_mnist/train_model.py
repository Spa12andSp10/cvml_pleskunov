import random
from PIL import Image
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import optim
from pathlib import Path
import os
import zipfile

save_path = Path(__file__).parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class WorkingWithFiles():
    def __init__(self, rate=0.75):
        self.directory = Path("./cyrillic/Cyrillic")
        self.train = Path("./train")
        self.test = Path("./test")
        self.rate = rate

    def open_zip_file(self):
        if 'cyrillic' not in os.listdir('./'):
            with zipfile.ZipFile('cyrillic.zip', 'r') as zip_ref:
                zip_ref.extractall('cyrillic')
        else:
            print("File is already in directory!")

    def directories(self, letter):
        trainp = self.train / letter
        testp = self.test / letter
        if not trainp.exists():
            os.mkdir(trainp)
        if not testp.exists():
            os.mkdir(testp)

    def make_train(self, file, id, letter):
        train = file[:id]
        for i in train:
            os.rename(str(self.directory / letter / i), str(self.train / letter / i))

    def make_test(self, file, id, letter):
        test = file[id:]
        for i in test:
            os.rename(str(self.directory / letter / i), str(self.test / letter / i))

    def tts(self):
        letters = os.listdir(self.directory)
        for l in letters:
            self.directories(l)
            f = os.listdir(str(self.directory / l))
            random.shuffle(f)
            id = int(len(f) * self.rate)
            self.make_train(f, id, l)
            self.make_test(f, id, l)

    def create(self):
        if "train" not in os.listdir("./"):
            os.mkdir("train")
        if "test" not in os.listdir("./"):
            os.mkdir("test")
        if len(os.listdir(self.test)) == 34 and len(os.listdir(self.train)) == 34:
            print("Data is already preprocessed!")
        else:
            self.open_zip_file()
            self.tts()


files = WorkingWithFiles()
files.create()


class MyDataset(Dataset):
    def __init__(self, train=False, transforms=None):
        if train:
            self.path = Path("./train")
        else:
            self.path = Path("./test")
        self.mylen = 0
        self.files = []
        self.cls = os.listdir(self.path)
        self.tr = transforms
        for label in self.cls:
            i = self.path / label
            list_files = os.listdir(i)
            self.mylen += len(list_files)
            for name in list_files:
                self.files.append((i / name, label))

    def __len__(self):
        return self.mylen

    def __getitem__(self, item):
        path_to_file, target = self.files[item]
        img = Image.open(path_to_file).split()[-1]
        label = self.cls.index(target)
        return self.tr(img), label


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomAffine(10, (0.1, 0.1), (0.5, 0.9), 10),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = MyDataset(train=True, transforms=transform)
test_dataset = MyDataset(train=False, transforms=transform_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


class CyrillicMNIST(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.selu1 = nn.SELU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.selu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.selu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.selu4 = nn.ELU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.selu5 = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.selu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.selu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.selu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.selu4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.selu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


model = CyrillicMNIST(num_classes=34).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 20
train_loss = []
train_acc = []

model_path = save_path / "model.pth"
if not model_path.exists():
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        epoch_loss = run_loss / len(train_loader)
        epoch_acc = 100 * (correct / total)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.3f}%")

    torch.save(model.cpu().state_dict(), model_path)
    model.to(device)
    print(f"Model saved to {model_path}")

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title("Training Loss")
    plt.plot(train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(122)
    plt.title("Training Accuracy")
    plt.plot(train_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.savefig(' train.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path} on {device}")
