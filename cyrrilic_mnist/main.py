from train_model import CyrillicMNIST, test_loader
import torch
from pathlib import Path

model_path = Path(__file__).parent / "model.pth"

if not model_path.exists():
    raise RuntimeError("Model not trained!")

model = CyrillicMNIST()
model.load_state_dict(torch.load(model_path))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.3f}%")