# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Fracture Detection AI",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image to detect fracture.")

# =========================================
# DEVICE
# =========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================
# CNN MODEL
# MUST MATCH TRAINING MODEL
# =========================================

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Prevents overfitting

            nn.Linear(512, len(train_dataset.classes))
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.fc(x)

        return x


# =========================================
# LOAD MODEL
# =========================================

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location=device
    )
)

model.eval()

# =========================================
# CLASS LABELS
# CHECK ORDER USING:
# print(train_dataset.class_to_idx)
# =========================================

classes = [
    "fractured",
    "normal"
]

# =========================================
# IMAGE TRANSFORMS
# MUST MATCH TRAINING
# =========================================

transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor()
])

# =========================================
# FILE UPLOADER
# =========================================

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png", "webp"]
)

# =========================================
# PREDICTION
# =========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[predicted.item()]

    confidence_score = confidence.item() * 100

    st.subheader("Prediction")

    if predicted_class == "fractured":

        st.error(
            f"🩻 Fracture Detected\n\n"
            f"Confidence: {confidence_score:.2f}%"
        )

    else:

        st.success(
            f"✅ No Fracture Detected\n\n"
            f"Confidence: {confidence_score:.2f}%"
        )

    st.subheader("All Probabilities")

    for i, cls in enumerate(classes):

        st.write(
            f"{cls}: "
            f"{probabilities[0][i].item() * 100:.2f}%"
        )
