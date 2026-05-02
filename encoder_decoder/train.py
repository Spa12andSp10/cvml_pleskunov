import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import torch.optim as optim

class ImageDataset(Dataset):

    def __init__(self, n = 200, size = 128, positionFix=True, fixText=True, fixLen=True):
        super().__init__()
        self.n = n
        self.size = size
        self.positionFix = positionFix
        self.fixText = fixText
        self.fixLen = fixLen
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size), color = 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        if self.positionFix == True:
            x = 30
            y = 30
        else:
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)
        if self.fixText == True:
            text = "ABC"
        else:
            if self.fixLen == True:
                cloud = ["ABC", "DEF", "HIJ", "KMN", "OPQ"]
                text = cloud[np.random.randint(0, len(cloud) - 1)]
            else:
                cloud = ["ABC", "DEFGHI", "JK", "KMNOPQRSTU", "V"]
                text = cloud[np.random.randint(0, len(cloud) - 1)]
        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor

ds = ImageDataset(2000, 256, False, False, False)
dataloader = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=8)
plt.imshow(ds[0][0][0])
plt.show()

class Encoder(nn.Module):

    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, stride=2, padding=1, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, stride=2, padding=1, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, stride=2, padding=1, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, stride=2, padding=1, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x

class Decoder(nn.Module):

    def __init__(self, latent_size = 512):
        super().__init__()
        self.bottleneck = nn.Linear(latent_size, 256*16*16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, stride=2, padding=1, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, stride=2, padding=1, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, stride=2, padding=1, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, stride=2, padding=1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x

encoder = Encoder()
decoder = Decoder()
# e = sum(p.numel() for p in en.parameters())
# d = sum(p.numel() for p in dec.parameters())
# print(e, d)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    encoder.to(device)
    decoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    encoder.train()
    decoder.train()
    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            latent = encoder(imgs)
            output = decoder(latent)
            loss = criterion(imgs, output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"{epoch=}, {avg_loss=:.2f}")

    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
