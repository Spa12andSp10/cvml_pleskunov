from unet_road import (RoadsDataset, model, path)
import torch
import matplotlib.pyplot as plt

model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

dataset = RoadsDataset(path)
for i in range(31):
    image, _ = dataset[i]
    with torch.no_grad():
        mask = model(image.unsqueeze(0))

        plt.subplot(131)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.subplot(132)
        plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        plt.subplot(133)
        diff = image.permute(1, 2, 0).cpu().numpy() - mask.squeeze().cpu().numpy()[:, :, None]
        plt.imshow(diff)
        plt.show()

