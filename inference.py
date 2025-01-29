import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import config

CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]

def predict(image_path, model_path="models/best_model.pth"):
    model = load_model(model_path)
    model.eval()
    
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_idx].item() * 100  # Convert to percentage
    
    # Return the class name and confidence
    return CLASS_NAMES[predicted_idx], confidence

# Example usage
if __name__ == "__main__":
    image_path = "data/sample.jpg"
    prediction, confidence = predict(image_path)
    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")