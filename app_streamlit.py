import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load the trained model
# ---------------------------
MODEL_PATH = "plant_disease_model.pth"  # Place this in the same folder
try:
    model = torch.load(MODEL_PATH, map_location=device)  # full model load
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------------------
# Class labels (must match model training)
# ---------------------------
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
    # Add other classes if your model has more
]

# ---------------------------
# Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # match training image size
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

def preprocess_image(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension

# ---------------------------
# Prediction function
# ---------------------------
def predict_plant_disease(image):
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌱 Plant Disease Prediction")

st.write("Upload an image of a leaf, and the model will predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        label = predict_plant_disease(image)
        st.success(f"Predicted Plant Disease: **{label}**")
