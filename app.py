import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ResNet50 model with pre-trained ImageNet weights
try:
    # Load pre-trained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Modify the final layer to match the number of classes (5 classes in your case)
    num_classes = 5  # Change this based on your dataset (e.g., 5 injury types)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Set the model to evaluation mode
    model.eval()

    logger.info("ResNet50 model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model.")

# Define class labels (update based on your dataset)
class_labels = ["Bruise", "Burn", "Cut", "Fracture", "Rash"]

# Define image transformations (resize, normalize, and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)  # Get the index of the highest probability
            result = class_labels[predicted.item()]  # Map the index to a label
        
        logger.info(f"Prediction: {result}")
        return {"prediction": result}
    
    except Exception as e:
        logger.error(f" Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run server: uvicorn app:app --host 127.0.0.1 --port 8000 --reload