import cv2
import time
import torch
from pathlib import Path
from train_model import predict, Buffer, train, transform, build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")
save_path = Path(__file__).parent
model_path = save_path / "model.pth"

model = build_model()
print(model)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
buffer = Buffer()
count_labeled = 0

while True:

    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if key == ord("q"):
        break

    elif key == ord("1"):
        tensor = transform(image)
        buffer.append(tensor, 1.0)
        count_labeled += 1
    elif key == ord("2"):
        tensor = transform(image)
        buffer.append(tensor, 0.0)
        count_labeled += 1
    elif key == ord("p"):
        t = time.perf_counter()
        label, confidence = predict(frame)
        print(f"Elapsed time {time.perf_counter() - t}")
        print(label, confidence)
    elif key == ord("s"):
        torch.save(model.state_dict(), model_path)
    if count_labeled >= buffer.frames.maxlen:
        loss = train(buffer)
        if loss:
            print(f"Loss = {loss}")
        count_labeled = 0