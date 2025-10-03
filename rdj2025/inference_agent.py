import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import io
import os

MODEL_PATH = "/home/game/camera_ws/src/rdj2025/models/potato_resnet18.pth"

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

print(" Checkpoint keys:", list(checkpoint.keys()))
print(" Model state dict keys (first few):", list(checkpoint["model_state_dict"].keys())[:10])

# Class names
if "class_names" in checkpoint:
    class_names = checkpoint["class_names"]
else:
    class_names = ["healthy", "early_blight", "late_blight"]

# Create ResNet18 backbone
model = models.resnet18(weights=None)

# ✅ Define a Sequential FC head (matching your trained model)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_names))
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

print(f" Model loaded successfully with custom FC head and classes: {class_names}")

# FastAPI setup
app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            prediction = class_names[pred.item()]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
