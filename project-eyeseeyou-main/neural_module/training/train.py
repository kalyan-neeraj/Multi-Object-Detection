# # neuro_symbolic_visual_reasoning/neural_module/training/train.py
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# import sys

# # Add parent directory to path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from config import Config
# from models.detector import create_model
# from utils.data_loader import get_data_loaders

# def train_model(config):
#     """
#     Train the object detector with attribute classification
    
#     Args:
#         config: Configuration object
#     """
#     # Set device
#     device = config.DEVICE
#     print(f"Using device: {device}")
    
#     # Create data loaders
#     train_loader, val_loader = get_data_loaders(config)
#     print(f"Train dataset size: {len(train_loader.dataset)}")
#     print(f"Validation dataset size: {len(val_loader.dataset)}")
    
#     # Create model
#     model = create_model(config)
#     model = model.to(device)
    
#     # Create optimizer
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.Adam(params, lr=config.LEARNING_RATE)
    
#     # Create scheduler for learning rate
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
#     # Create directory for saving models
#     os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
#     # Training loop
#     best_val_loss = float('inf')
    
#     for epoch in range(config.NUM_EPOCHS):
#         print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
#         # Training phase
#         model.train()
#         train_loss = 0.0
        
#         for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
#             # Move batch to device
#             images = [img.to(device) for img in batch["images"]]
            
#             # Prepare targets with correct format for Faster R-CNN
#             targets = []
#             for i in range(len(images)):
#                 boxes = batch["boxes"][i]
#                 if len(boxes) > 0:
#                     target = {
#                         "boxes": boxes.to(device),
#                         # All objects are class 1 (foreground)
#                         "labels": torch.ones(len(boxes), dtype=torch.int64).to(device)
#                     }
#                     targets.append(target)
#                 else:
#                     # Skip images with no boxes
#                     continue
            
#             # Skip batch if no valid targets
#             if len(targets) == 0:
#                 continue
            
#             # Zero the gradients
#             optimizer.zero_grad()
            
#             # Forward pass
#             loss_dict = model(images, targets)
            
#             # Compute total loss
#             losses = sum(loss for loss in loss_dict.values())
            
#             # Backward pass
#             losses.backward()
            
#             # Update weights
#             optimizer.step()
            
#             # Update running loss
#             train_loss += losses.item()
            
#             # Print interim results
#             if batch_idx % 50 == 0:
#                 print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")
        
#         # Calculate average training loss
#         train_loss /= len(train_loader)
#         print(f"Training loss: {train_loss:.4f}")
        
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
        
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validation"):
#                 # Move batch to device
#                 images = [img.to(device) for img in batch["images"]]
                
#                 # Prepare targets with correct format for Faster R-CNN
#                 targets = []
#                 for i in range(len(images)):
#                     boxes = batch["boxes"][i]
#                     if len(boxes) > 0:
#                         target = {
#                             "boxes": boxes.to(device),
#                             # All objects are class 1 (foreground)
#                             "labels": torch.ones(len(boxes), dtype=torch.int64).to(device)
#                         }
#                         targets.append(target)
#                     else:
#                         # Skip images with no boxes
#                         continue
                
#                 # Skip batch if no valid targets
#                 if len(targets) == 0:
#                     continue
                
#                 # Forward pass
#                 loss_dict = model(images, targets)
                
#                 # Compute total loss
#                 losses = sum(loss for loss in loss_dict.values())
                
#                 # Update running loss
#                 val_loss += losses.item()
        
#         # Calculate average validation loss
#         val_loss /= len(val_loader)
#         print(f"Validation loss: {val_loss:.4f}")
        
#         # Update learning rate
#         scheduler.step()
        
#         # Save model if validation loss improved
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             model_path = os.path.join(config.MODEL_SAVE_DIR, f"detector_epoch_{epoch+1}.pth")
#             torch.save(model.state_dict(), model_path)
#             print(f"Model saved to {model_path}")

# if __name__ == "__main__":
#     config = Config()
#     train_model(config)

import os
import sys
import torch
from pathlib import Path

# Add project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_module.config import Config
from neural_module.models.detector import create_model
from neural_module.utils.data_loader import prepare_yolo_dataset

def train_model(config):
    """Train a YOLO model on CLEVR dataset"""
    print(f"Using device: {config.DEVICE}")
    
    # Prepare dataset in YOLO format
    yaml_path = prepare_yolo_dataset(config)
    print(f"Dataset prepared with YAML config at {yaml_path}")
    
    # Create and initialize model
    model = create_model(config)
    
    # Start training
    print(f"Starting YOLOv8l training for {config.NUM_EPOCHS} epochs...")
    results = model.train_model(
        data_yaml=yaml_path,
        epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        imgsz=config.IMAGE_SIZE
    )
    
    # Print training results
    print("Training completed!")
    print(f"Best model saved at: {os.path.join(config.MODEL_SAVE_DIR, 'yolo_clevr/weights/best.pt')}")
    
    # Copy best model to config location
    best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'yolo_clevr/weights/best.pt')
    if os.path.exists(best_model_path):
        import shutil
        dest_path = os.path.join(config.MODEL_SAVE_DIR, 'detector_best.pt')
        shutil.copy2(best_model_path, dest_path)
        print(f"Best model copied to {dest_path}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLOv8l model on CLEVR dataset')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--size', type=int, default=None, help='Image size')
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch:
        config.BATCH_SIZE = args.batch
    if args.size:
        config.IMAGE_SIZE = args.size
    
    # Train model
    train_model(config)
