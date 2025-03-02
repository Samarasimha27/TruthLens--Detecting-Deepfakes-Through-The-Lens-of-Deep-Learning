import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import dlib
import torch.nn as nn
from torchvision import models
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Define the EnsembleModel class
class EnsembleModel(nn.Module):
    def __init__(self, resnet, efficientnet, mobilenet):
        super(EnsembleModel, self).__init__()
        self.resnet = resnet
        self.efficientnet = efficientnet
        self.mobilenet = mobilenet
        self.fc = nn.Linear(2048 + 1280 + 1280, 2)

    def forward(self, x):
        resnet_features = self.resnet(x)
        efficientnet_features = self.efficientnet(x)
        mobilenet_features = self.mobilenet(x)
        combined_features = torch.cat((resnet_features, efficientnet_features, mobilenet_features), dim=1)
        output = self.fc(combined_features)
        return output

# Load pre-trained models
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

resnet.fc = nn.Identity()
efficientnet.classifier[1] = nn.Identity()
mobilenet.classifier[1] = nn.Identity()

# Unfreeze specific layers
for model in [resnet, efficientnet, mobilenet]:
    for name, param in model.named_parameters():
        if "layer4" in name or "blocks" in name or "features" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

resnet = resnet.to(device)
efficientnet = efficientnet.to(device)
mobilenet = mobilenet.to(device)

# Initialize Ensemble Model
ensemble_model = EnsembleModel(resnet, efficientnet, mobilenet).to(device)

# Define the checkpoint path
checkpoint_path = r"C:\Users\91970\Desktop\TruthLens_Deepfake_Detector\ensemble_model_state.pth"

# Check if the file exists
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'. Please ensure the file exists.")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Debug checkpoint keys
# st.write("Checkpoint Keys:", list(checkpoint.keys()))

# Load model weights
if 'model_state_dict' in checkpoint:
    ensemble_model.load_state_dict(checkpoint['model_state_dict'])
else:
    ensemble_model.load_state_dict(checkpoint)

ensemble_model.eval()

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function for face detection and cropping
def detect_and_crop_face(image):
    # Ensure the image is in RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image, 1)

    if len(faces) > 0:
        # Get the coordinates of the first detected face
        x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
        x, y = max(0, x), max(0, y)  # Ensure within bounds
        cropped_image = rgb_image[y:y+h, x:x+w]
    else:
        st.warning("No face detected. Please try another image.")
        return None

    # Convert the cropped image to PIL format and apply transformations
    cropped_image = Image.fromarray(cropped_image)
    return transform(cropped_image).unsqueeze(0)

# Inference function
def predict(image):
    input_tensor = detect_and_crop_face(image)
    if input_tensor is None:
        return None, None
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = ensemble_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
        class_names = ['Fake', 'Real']
        return class_names[predicted.item()], probabilities[0].cpu().numpy()

# Streamlit app
st.title("TruthLens : Deepfake Detector")
st.write("Upload an image, and the model will classify it as Real or Fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Convert PIL image to OpenCV format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    prediction, probabilities = predict(image_cv2)

    if prediction:
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Probabilities:** Fake: {probabilities[0]:.3f}, Real: {probabilities[1]:.3f}")

# Instructions for running Streamlit app
# cd C:\Users\91970\Desktop\TruthLens_Deepfake_Detector
# streamlit run TruthLens_app.py
