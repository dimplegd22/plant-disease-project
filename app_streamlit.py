import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io

# Title
st.title("🌿 Plant Disease Detection App")
import os
import gdown

MODEL_PATH = "plant_disease_model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1PsDJwg5L45i5e60la8xjS-4v7YFWs2RA"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
except Exception as e:
    import streamlit as st
    st.error(f"Model loading failed: {e}")
import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mode

CLASS_LABELS = [
    'Apple___Apple_scab',
    'Apple___healthy',
    'Tomato___Early_blight',
    'Tomato___healthy'
]

class PlantDiseaseCNN(torch.nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 127 * 127, len(CLASS_LABELS))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

device = torch.device("cpu")
model = PlantDiseaseCNN()

model.eval()

# Image preprocess
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Prediction
def predict(image):
    img = preprocess_image(image)
    with torch.no_grad():
        output = model(img)
        index = torch.argmax(output, dim=1).item()
        return CLASS_LABELS[index]

# Upload image
uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict(image)
        st.success(f"Prediction: {result}")
