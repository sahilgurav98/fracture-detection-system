import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="Fracture Detection",
    layout="centered"
)

st.title("Bone Fracture Detection")

device = torch.device("cpu")

predict_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                2
            )
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.fc(x)

        return x

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location=device
    )
)

model.eval()

classes = [
    "fractured",
    "not fractured"
]

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        width=400
    )

    image_tensor = predict_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    TEMPERATURE = 2.0

    with torch.no_grad():

        output = model(image_tensor)

        scaled_output = output / TEMPERATURE

        probabilities = torch.softmax(
            scaled_output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[
        predicted.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )

    st.subheader("Prediction")

    st.write(predicted_class)

    st.write(
        f"Confidence: "
        f"{confidence_score:.2f}%"
    )

    st.subheader("Probabilities")

    for i, cls in enumerate(classes):

        prob = (
            probabilities[0][i].item()
            * 100
        )

        st.write(
            f"{cls}: {prob:.2f}%"
        )
