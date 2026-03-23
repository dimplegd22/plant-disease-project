import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Define class labels manually
# (matches the classes your model was trained on)
# ---------------------------
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
    # add more classes if your model has them
]

# ---------------------------
# Model definition (matches training)
# ---------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=len(class_labels)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flatten size using dummy input (128x128)
        dummy_input = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.pool(torch.relu(self.conv4(x)))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, self.flattened_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Load the trained model
# ---------------------------
MODEL_PATH = "plant_disease_model.pth"  # make sure this file is in the same folder
model = PlantDiseaseCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# ---------------------------
# Preprocessing for inference
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # must match training size
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # same as training
])

# ---------------------------
# Prediction functions
# ---------------------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension

def predict_plant_disease(image_path):
    img_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    test_image = "sample_apple_scab.jpg"  # place a test image in the same folder
    label = predict_plant_disease(test_image)
    print("Predicted Plant Disease:", label)
