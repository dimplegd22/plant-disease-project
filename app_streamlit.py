import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import gdown

# ---------------------------
# ✅ CLASS LABELS (EDIT if needed)
# ---------------------------
CLASS_LABELS = [
    "Healthy",
    "Bacterial Spot",
    "Early Blight",
    "Late Blight"
]

# ---------------------------
# ✅ MODEL ARCHITECTURE
# ---------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN, self).__init__()

        # MATCHING YOUR MODEL
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)

        # Fully connected
        self.fc1 = nn.Linear(57600, 512)
        self.fc2 = nn.Linear(512, len(CLASS_LABELS))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ---------------------------
# ✅ DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# ✅ CREATE MODEL
# ---------------------------
model = PlantDiseaseCNN()

# ---------------------------
# ✅ MODEL PATH
# ---------------------------
MODEL_PATH = "plant_disease_model.pth"

# ---------------------------
# ✅ DOWNLOAD MODEL FROM GOOGLE DRIVE
# ---------------------------
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1PsDJwg5L45i5e60la8xjS-4v7YFWs2RA"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# ---------------------------
# ✅ LOAD MODEL
# ---------------------------
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"Model loading failed: {e}")

# ---------------------------
# ✅ IMAGE TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------------------------
# ✅ STREAMLIT UI
# ---------------------------
st.title("🌿 Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        result = CLASS_LABELS[predicted.item()]

    st.success(f"Prediction: {result}")
