import argparse
import torch
from src.model import Net
from dataset_loader import trainloader, validloader
from validate import validate
from inference import predict
from train import train
import config

def main():
    parser = argparse.ArgumentParser(description='Train, Validate, or Predict using your model.')
    parser.add_argument('mode', choices=['train', 'validate', 'predict'], help="Mode to run: train, validate, or predict")
    parser.add_argument('--model', type=str, help="Path to model file (e.g., model.pth) for validation or prediction")
    parser.add_argument('--image', type=str, help="Path to image file for prediction (only required in predict mode)")

    args = parser.parse_args()

    if args.mode == 'train':
        train(config.model, config.criterion, config.optimizer)

    elif args.mode == 'validate':
        if not args.model:
            print("Error: Please provide a model file path for validation using --model")
            return
        model = Net()
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        model.eval()
        validate(model, validloader, config.criterion)

    # Predict mode
    elif args.mode == 'predict':
        if not args.model or not args.image:
            print("Error: Please provide both --model (model path) and --image (image path) for prediction")
            return
        model = Net()
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        model.eval()
        
        class_name, confidence = predict(model, args.image)
        print(f"Prediction: {class_name}, Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
