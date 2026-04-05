import torch
import torchvision.models
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")
save_path = Path(__file__).parent
model_path = save_path / "model.pth"
print(save_path)

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(features, 1)
    )
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    return model.to(device)

# def build_model():
#     weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
#     model = torchvision.models.alexnet(weights=weights)
#     for param in model.features.parameters():
#         param.requires_grad = False
#
#     features = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(features, 1)
#     if model_path.exists():
#         model.load_state_dict(torch.load(model_path))
#     return model.to(device)

model = build_model()
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train(buffer):
    if len(buffer) < 10:
        return None
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    prediction = model(images).squeeze(1)
    loss = criterion(prediction, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = "person" if prob > 0.5 else "no person"
    return label, prob

class Buffer():
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def get_batch(self):
        images = torch.stack(list(self.frames)).to(device)
        labels = torch.tensor(list(self.labels), dtype=torch.float32).to(device)
        return images, labels



# cap = cv2.VideoCapture(0)
# cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
# buffer = Buffer()
# count_labeled = 0
#
# while True:
#
#     _, frame = cap.read()
#     cv2.imshow("Camera", frame)
#     key = cv2.waitKey(1) & 0xFF
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     if key == ord("q"):
#         break
#
#     elif key == ord("1"):
#         tensor = transform(image)
#         buffer.append(tensor, 1.0)
#         count_labeled += 1
#     elif key == ord("2"):
#         tensor = transform(image)
#         buffer.append(tensor, 0.0)
#         count_labeled += 1
#     elif key == ord("p"):
#         t = time.perf_counter()
#         label, confidence = predict(frame)
#         print(f"Elapsed time {time.perf_counter() - t}")
#         print(label, confidence)
#     elif key == ord("s"):
#         torch.save(model.state_dict(), model_path)
#     if count_labeled >= buffer.frames.maxlen:
#         loss = train(buffer)
#         if loss:
#             print(f"Loss = {loss}")
#         count_labeled = 0
