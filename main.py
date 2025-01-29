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
    parser.add_argument('--model', type=str, help="Path to model file (e.g., model.pth) for prediction or validation")
    
    args = parser.parse_args()


    # Train mode
    if args.mode == 'train':
        train(config.model, trainloader, validloader, config.criterion, config.optimizer)
    
    # Validate mode
    elif args.mode == 'validate':
        if not args.model:
            print("Error: Please provide a model file path for validation using --model")
            return
        model = torch.load(args.model)
        validate(model, validloader, config.criterion)
    
    # Predict mode
    elif args.mode == 'predict':
        if not args.model:
            print("Error: Please provide a model file path for prediction using --model")
            return
        model = torch.load(args.model)
        predict(model, validloader)

if __name__ == "__main__":
    main()
