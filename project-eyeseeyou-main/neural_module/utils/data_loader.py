# # neuro_symbolic_visual_reasoning/neural_module/utils/data_loader.py
# import os
# import json
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as transforms
# import numpy as np

# class CLEVRObjectDataset(Dataset):
#     def __init__(self, scenes_file, images_dir, split_file, split="train", transform=None):
#         """
#         Dataset for object detection and attribute classification on CLEVR
        
#         Args:
#             scenes_file: Path to scene JSON file
#             images_dir: Directory containing images
#             split_file: Path to train/val split file
#             split: 'train' or 'val'
#             transform: Torchvision transforms
#         """
#         self.images_dir = images_dir
        
#         # Load scene data
#         with open(scenes_file, 'r') as f:
#             self.scenes_data = json.load(f)
        
#         # Load split information
#         with open(split_file, 'r') as f:
#             split_data = json.load(f)
        
#         # Get filenames for the requested split
#         self.image_filenames = split_data[split]
        
#         # Create a mapping from image filenames to scene data
#         self.filename_to_scene = {
#             scene["image_filename"]: scene
#             for scene in self.scenes_data["scenes"]
#         }
        
#         # Set up transforms
#         if transform is None:
#             self.transform = transforms.Compose([
#                 transforms.Resize((320, 480)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             ])
#         else:
#             self.transform = transform
    
#     def __len__(self):
#         return len(self.image_filenames)
    
#     def __getitem__(self, idx):
#         # Get image filename
#         img_filename = self.image_filenames[idx]
        
#         # Load image
#         img_path = os.path.join(self.images_dir, img_filename)
#         image = Image.open(img_path).convert("RGB")
        
#         # Get scene data for this image
#         scene = self.filename_to_scene[img_filename]
#         objects = scene["objects"]
        
#         # Prepare object data: boxes, colors, shapes, materials, sizes
#         num_objects = len(objects)
        
#         # Normalize coordinates to [0, 1]
#         boxes = []
#         colors = []
#         shapes = []
#         materials = []
#         sizes = []
        
#         # Image dimensions (CLEVR images are 480x320)
#         img_width, img_height = 480, 320
        
#         for obj in objects:
#             # Get 3D coordinates
#             x, y, z = obj["3d_coords"]
            
#             # Approximate 2D bounding box from 3D coordinates
#             # CLEVR specific conversion - this is approximate and would need refinement
#             # in a real implementation with more precise object localization
            
#             # Convert x, z coordinates to pixel space
#             # This is a simplification - proper projection would be better
#             center_x = (x + 3) / 6 * img_width  # x in [-3, 3]
#             center_y = (1 - (z + 1) / 4) * img_height  # z in [-1, 3]
            
#             # Size based box dimensions - rough approximation
#             size_factor = 0.15 if obj["size"] == "large" else 0.1
#             box_w = size_factor * img_width
#             box_h = size_factor * img_height
            
#             # Create box [x1, y1, x2, y2]
#             box = [
#                 max(0, center_x - box_w/2),
#                 max(0, center_y - box_h/2),
#                 min(img_width, center_x + box_w/2),
#                 min(img_height, center_y + box_h/2)
#             ]
            
#             # Ensure boxes have non-zero area
#             if box[2] > box[0] and box[3] > box[1]:
#                 boxes.append(box)
#                 colors.append(obj["color"])
#                 shapes.append(obj["shape"])
#                 materials.append(obj["material"])
#                 sizes.append(obj["size"])
        
#         # Apply transforms to image
#         image_tensor = self.transform(image)
        
#         # Convert boxes to tensor
#         if boxes:
#             boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
#         else:
#             # Create an empty tensor with shape [0, 4] if no boxes
#             boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        
#         return {
#             "image": image_tensor,
#             "image_filename": img_filename,
#             "boxes": boxes_tensor,
#             "colors": colors,
#             "shapes": shapes,
#             "materials": materials,
#             "sizes": sizes,
#             "num_objects": len(boxes)  # This might be different from num_objects if we filtered some boxes
#         }

# def collate_fn(batch):
#     """
#     Custom collate function to handle variable number of objects per image
#     """
#     images = [item["image"] for item in batch]  # List of image tensors
#     filenames = [item["image_filename"] for item in batch]
    
#     # Variable length data
#     all_boxes = [item["boxes"] for item in batch]
#     all_colors = [item["colors"] for item in batch]
#     all_shapes = [item["shapes"] for item in batch]
#     all_materials = [item["materials"] for item in batch]
#     all_sizes = [item["sizes"] for item in batch]
    
#     num_objects = [item["num_objects"] for item in batch]
    
#     return {
#         "images": images,  # List of tensors, not stacked
#         "image_filenames": filenames,
#         "boxes": all_boxes,
#         "colors": all_colors,
#         "shapes": all_shapes,
#         "materials": all_materials,
#         "sizes": all_sizes,
#         "num_objects": num_objects
#     }

# def get_data_loaders(config):
#     """
#     Create train and validation data loaders
    
#     Args:
#         config: Configuration object
    
#     Returns:
#         train_loader, val_loader
#     """
#     # Create datasets
#     train_dataset = CLEVRObjectDataset(
#         scenes_file=config.TRAIN_SCENES,
#         images_dir=os.path.join(config.DATA_DIR, "images"),
#         split_file=config.TRAIN_VAL_SPLIT,
#         split="train"
#     )
    
#     val_dataset = CLEVRObjectDataset(
#         scenes_file=config.TRAIN_SCENES,
#         images_dir=os.path.join(config.DATA_DIR, "images"),
#         split_file=config.TRAIN_VAL_SPLIT,
#         split="val"
#     )
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=4
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=4
#     )
    
#     return train_loader, val_loader


import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import yaml
from pathlib import Path

class CLEVRObjectDataset(Dataset):
    def __init__(self, scenes_file, images_dir, split_file, split="train", transform=None, image_size=640):
        self.images_dir = images_dir
        self.image_size = image_size
        
        # Load scene data
        with open(scenes_file, 'r') as f:
            self.scenes_data = json.load(f)
        
        # Load split data
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        self.image_filenames = split_data[split]
        
        # Create mapping from filename to scene
        self.filename_to_scene = {
            scene["image_filename"]: scene
            for scene in self.scenes_data["scenes"]
            if scene["image_filename"] in self.image_filenames
        }
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get image filename
        img_filename = self.image_filenames[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        
        # Get scene data
        scene = self.filename_to_scene[img_filename]
        objects = scene["objects"]
        
        # Process image
        img = self.transform(image)
        
        # Process bounding boxes and labels
        boxes = []
        class_ids = []  # 0 for all objects (single class)
        
        # For YOLO: Process objects to get normalized boxes in x_center, y_center, width, height format
        for obj in objects:
            x, y, z = obj["3d_coords"]
            
            # Convert to 2D center point with proper scaling
            center_x = (x + 3) / 6  # x in [-3, 3] -> [0, 1]
            center_y = (1 - (z + 1) / 4)  # z in [-1, 3] -> [1, 0]
            
            # Calculate depth factor for object scaling
            depth_factor = 4.5 / (z + 4.5)  # Gentler depth scaling
            
            # Adjust box size based on object shape and size
            if obj["shape"] == "cube":
                width = height = 0.20 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
            elif obj["shape"] == "sphere":
                width = height = 0.19 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
            elif obj["shape"] == "cylinder":
                width = 0.17 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
                height = 0.22 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
            else:
                # Default for unknown shapes
                width = height = 0.20 * depth_factor
            
            # Apply material-specific adjustment
            if obj["material"] == "metal":
                # Metal objects often have reflections
                width *= 1.1
                height *= 1.1
            
            # Ensure the box fits within the image
            if 0 < center_x < 1 and 0 < center_y < 1 and width > 0 and height > 0:
                boxes.append([center_x, center_y, width, height])
                class_ids.append(0)  # Single class for all objects
        
        # Convert to tensors
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(class_ids, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
        
        # Return data in format expected by YOLOv8
        return {
            "img": img,
            "img_path": img_path,
            "ori_shape": (orig_height, orig_width),
            "img_size": (self.image_size, self.image_size),
            "bboxes": boxes_tensor,
            "cls": labels_tensor,
        }

def create_yolo_yaml(config):
    """Create a YAML file for YOLOv8 training configuration"""
    
    # Get absolute paths to avoid path duplication
    base_dir = os.path.abspath(config.DATA_DIR)
    
    data = {
        'path': base_dir,
        'train': os.path.join(base_dir, 'images/train'),
        'val': os.path.join(base_dir, 'images/val'),
        
        'names': {
            0: 'object'  # Single class for all CLEVR objects
        },
        
        'nc': 1  # Number of classes
    }
    
    # Write the YAML file
    yaml_path = os.path.abspath(config.YAML_PATH)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Created YAML config at {yaml_path} with paths:")
    print(f"  Train: {data['train']}")
    print(f"  Val: {data['val']}")
    
    return yaml_path

def prepare_yolo_dataset(config):
    """Prepare dataset in YOLO format with improved bounding boxes"""
    
    # Get absolute path
    base_dir = os.path.abspath(config.DATA_DIR)
    
    # Create directories for YOLO format
    train_img_dir = os.path.join(base_dir, 'images/train')
    val_img_dir = os.path.join(base_dir, 'images/val')
    train_label_dir = os.path.join(base_dir, 'labels/train')
    val_label_dir = os.path.join(base_dir, 'labels/val')
    
    # Create directories if they don't exist
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    print(f"Created directories for YOLO dataset:")
    print(f"  Train images: {train_img_dir}")
    print(f"  Val images: {val_img_dir}")
    print(f"  Train labels: {train_label_dir}")
    print(f"  Val labels: {val_label_dir}")
    
    # Load train/val split
    with open(config.TRAIN_VAL_SPLIT, 'r') as f:
        split_data = json.load(f)
    
    # Load scenes
    with open(config.TRAIN_SCENES, 'r') as f:
        scenes_data = json.load(f)
    
    # Create mapping from filename to scene
    filename_to_scene = {
        scene["image_filename"]: scene
        for scene in scenes_data["scenes"]
    }
    
    # Process train and val sets
    objects_processed = 0
    
    for split in ['train', 'val']:
        images_processed = 0
        
        for img_filename in split_data[split]:
            # Skip if scene data not available
            if img_filename not in filename_to_scene:
                continue
            
            # Get scene data
            scene = filename_to_scene[img_filename]
            objects = scene["objects"]
            
            # Source image path
            src_path = os.path.join(base_dir, 'images', img_filename)
            
            # Skip if source doesn't exist
            if not os.path.exists(src_path):
                continue
                
            # Get image dimensions for better scaling
            try:
                with Image.open(src_path) as img:
                    img_width, img_height = img.size
            except:
                # Default size if image can't be opened
                img_width, img_height = 480, 320
            
            # Create destination paths
            dst_img_dir = train_img_dir if split == 'train' else val_img_dir
            dst_path = os.path.join(dst_img_dir, img_filename)
            
            # Create label file path
            label_dir = train_label_dir if split == 'train' else val_label_dir
            label_path = os.path.join(
                label_dir, 
                Path(img_filename).stem + '.txt'
            )
            
            # Create symbolic link or copy file
            if not os.path.exists(dst_path):
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    # If symlink fails, copy the file
                    import shutil
                    shutil.copy2(src_path, dst_path)
            
            # Track number of objects for this image
            image_object_count = 0
            
            # Generate YOLO format labels with improved bounding boxes
            with open(label_path, 'w') as f:
                for obj in objects:
                    x, y, z = obj["3d_coords"]
                    
                    # Convert to normalized coordinates
                    center_x = (x + 3) / 6  # x in [-3, 3] -> [0, 1]
                    center_y = (1 - (z + 1) / 4)  # z in [-1, 3] -> [1, 0]
                    
                    # Improved depth factor calculation
                    depth_factor = 4.5 / (z + 4.5)  # Gentler depth scaling
                    
                    # Shape-specific adjustments with larger boxes
                    if obj["shape"] == "cube":
                        width = height = 0.20 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
                    elif obj["shape"] == "sphere":
                        width = height = 0.19 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
                    elif obj["shape"] == "cylinder":
                        width = 0.17 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
                        height = 0.22 * depth_factor * (1.4 if obj["size"] == "large" else 0.9)
                    else:
                        # Default for unknown shapes
                        width = height = 0.20 * depth_factor
                    
                    # Apply material-specific adjustment (metallic objects often have reflections)
                    if obj["material"] == "metal":
                        width *= 1.1
                        height *= 1.1
                    
                    # Safety check: ensure the box fits within the image
                    if (center_x > 0 and center_x < 1 and 
                        center_y > 0 and center_y < 1 and 
                        width > 0 and height > 0):
                        
                        # Clamp values to valid range for YOLO
                        center_x = max(0.01, min(0.99, center_x))
                        center_y = max(0.01, min(0.99, center_y))
                        width = min(width, 0.5)  # Prevent too large boxes
                        height = min(height, 0.5)
                        
                        # Write in YOLO format: class x_center y_center width height
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        image_object_count += 1
            
            # Count processed objects
            objects_processed += image_object_count
            images_processed += 1
            
            # If no objects found in image, generate empty label file to avoid errors
            if image_object_count == 0:
                with open(label_path, 'w') as f:
                    pass
    
    print(f"Processed {objects_processed} objects across {images_processed} images for YOLO training")
    
    # Create YAML file
    yaml_path = create_yolo_yaml(config)
    
    return yaml_path

def get_data_loaders(config):
    """Legacy function for compatibility with existing code"""
    print("Note: For YOLO training, use prepare_yolo_dataset instead of get_data_loaders")
    
    train_dataset = CLEVRObjectDataset(
        scenes_file=config.TRAIN_SCENES,
        images_dir=os.path.join(config.DATA_DIR, "images"),
        split_file=config.TRAIN_VAL_SPLIT,
        split="train",
        image_size=config.IMAGE_SIZE
    )
    
    val_dataset = CLEVRObjectDataset(
        scenes_file=config.TRAIN_SCENES,
        images_dir=os.path.join(config.DATA_DIR, "images"),
        split_file=config.TRAIN_VAL_SPLIT,
        split="val",
        image_size=config.IMAGE_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader
