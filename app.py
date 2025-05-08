import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Step 1: Define the model architecture (make sure it matches the trained model)
class MaskDetectorModel(nn.Module):
    def __init__(self):
        super(MaskDetectorModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 1),  # Adjust this line to your architecture
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Step 2: Load the trained model
model = MaskDetectorModel()
model.load_state_dict(torch.load(r"E:\tensorflow\mask_detector_model.h5", map_location=torch.device('cpu')))
model.eval()

# Step 3: Prepare image
image_path = r"E:\tensorflow\WhatsApp Image 2025-04-10 at 9.06.21 PM.jpeg"
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),               # Converts to [0,1] and channels first
    transforms.Normalize([0.5]*3, [0.5]*3)  # Optional: use based on training
])

image = transform(image).unsqueeze(0)  # Add batch dimension

# Step 4: Predict
with torch.no_grad():
    prediction = model(image)
    prediction = torch.sigmoid(prediction)  # If not already applied in model
    pred_val = prediction.item()

# Step 5: Output label
label = "Mask" if pred_val < 0.5 else "No Mask"
print("Raw prediction:", pred_val)
print("Predicted Label:", label)
