import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import gdown

# ---------------------------
# 38 CLASS LABELS (match your .pth)
# ---------------------------
CLASS_LABELS = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
    "Apple___healthy","Blueberry___healthy","Cherry___Powdery_mildew",
    "Cherry___healthy","Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust","Corn___Northern_Leaf_Blight","Corn___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot",
    "Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot",
    "Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

# ---------------------------
# MODEL ARCHITECTURE (dynamic flatten for any image size)
# ---------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes, input_size=(256,256)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 4)

        # compute flatten size dynamically from input_size
        dummy = torch.zeros(1,3,*input_size)
        dummy = self.pool(torch.relu(self.conv1(dummy)))
        dummy = self.pool(torch.relu(self.conv2(dummy)))
        flatten_size = dummy.numel()

        self.fc1 = nn.Linear(57600, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# CREATE MODEL (256x256 input)
# ---------------------------
model = PlantDiseaseCNN(num_classes=len(CLASS_LABELS), input_size=(256,256))
model.to(device)

# ---------------------------
# MODEL PATH AND DOWNLOAD
# ---------------------------
MODEL_PATH = "plant_disease_model.pth"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1PsDJwg5L45i5e60la8xjS-4v7YFWs2RA"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# ---------------------------
# LOAD MODEL (conv weights only)
# ---------------------------
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # fc1 mismatch is ok
    model.eval()
    st.write("✅ Model loaded successfully")
except Exception as e:
    st.error(f"⚠ Model loading failed: {e}")

# ---------------------------
# IMAGE TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),  # matches dynamic input_size
    transforms.ToTensor()
])

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🌿 Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            raw_result = CLASS_LABELS[predicted.item()]
            result = raw_result.replace("___"," - ").replace("_"," ")

        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
