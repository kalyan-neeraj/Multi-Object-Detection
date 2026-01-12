# # # # neuro_symbolic_visual_reasoning/neural_module/models/detector.py
# # # import torch
# # # import torch.nn as nn
# # # import torchvision
# # # from torchvision.models.detection import FasterRCNN
# # # from torchvision.models.detection.rpn import AnchorGenerator

# # # class AttributeHead(nn.Module):
# # #     """
# # #     Custom head for attribute classification (color, shape, material, size)
# # #     """
# # #     def __init__(self, in_features, num_colors, num_shapes, num_materials, num_sizes):
# # #         super().__init__()
# # #         self.common_fc = nn.Sequential(
# # #             nn.Linear(in_features, 512),
# # #             nn.ReLU(),
# # #             nn.Dropout(0.3),
# # #         )
        
# # #         # Separate prediction heads for each attribute
# # #         self.color_fc = nn.Linear(512, num_colors)
# # #         self.shape_fc = nn.Linear(512, num_shapes)
# # #         self.material_fc = nn.Linear(512, num_materials)
# # #         self.size_fc = nn.Linear(512, num_sizes)
    
# # #     def forward(self, x):
# # #         x = self.common_fc(x)
# # #         return {
# # #             'color': self.color_fc(x),
# # #             'shape': self.shape_fc(x),
# # #             'material': self.material_fc(x),
# # #             'size': self.size_fc(x)
# # #         }

# # # class CLEVRObjectDetector(nn.Module):
# # #     """
# # #     Object detector for CLEVR dataset with attribute classification
# # #     """
# # #     def __init__(self, config):
# # #         super().__init__()
# # #         self.config = config
        
# # #         # Load backbone
# # #         if config.BACKBONE == "mobilenet_v2":
# # #             backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# # #             backbone_out = 1280
# # #         elif config.BACKBONE == "resnet18":
# # #             backbone = torchvision.models.resnet18(pretrained=True)
# # #             backbone_out = 512
# # #             # Remove the last layers
# # #             backbone = nn.Sequential(*list(backbone.children())[:-2])
# # #         elif config.BACKBONE == "efficientnet_b0":
# # #             backbone = torchvision.models.efficientnet_b0(pretrained=True).features
# # #             backbone_out = 1280
# # #         else:
# # #             raise ValueError(f"Unsupported backbone: {config.BACKBONE}")
        
# # #         # Freeze early layers for transfer learning
# # #         for param in list(backbone.parameters())[:-10]:
# # #             param.requires_grad = False
            
# # #         # Feature dimensions
# # #         backbone_out_channels = backbone_out
        
# # #         # RPN anchor generator
# # #         anchor_generator = AnchorGenerator(
# # #             sizes=((32, 64, 128, 256),),
# # #             aspect_ratios=((0.5, 1.0, 2.0),)
# # #         )
        
# # #         # ROI pooler
# # #         roi_pooler = torchvision.ops.MultiScaleRoIAlign(
# # #             featmap_names=['0'],
# # #             output_size=7,
# # #             sampling_ratio=2
# # #         )
        
# # #         # Create box predictor
# # #         box_predictor = nn.Linear(config.FEATURE_DIM, 4)
        
# # #         # Create attribute head
# # #         attribute_head = AttributeHead(
# # #             config.FEATURE_DIM, 
# # #             len(config.COLORS),
# # #             len(config.SHAPES),
# # #             len(config.MATERIALS),
# # #             len(config.SIZES)
# # #         )
        
# # #         # Use Faster R-CNN framework but customize it for our task
# # #         self.model = FasterRCNN(
# # #             backbone,
# # #             num_classes=2,  # Background and object
# # #             rpn_anchor_generator=anchor_generator,
# # #             box_roi_pool=roi_pooler,
# # #             box_head=nn.Sequential(
# # #                 nn.Linear(backbone_out_channels * 7 * 7, config.FEATURE_DIM),
# # #                 nn.ReLU(),
# # #                 nn.Dropout(0.3)
# # #             ),
# # #             box_predictor=box_predictor
# # #         )
        
# # #         # Add our custom attribute head
# # #         self.attribute_head = attribute_head
    
# # #     def forward(self, images, targets=None):
# # #         if self.training:
# # #             # During training, we need to format targets to Faster R-CNN format
# # #             formatted_targets = self._format_targets(targets)
# # #             losses = self.model(images, formatted_targets)
            
# # #             # Add attribute classification loss
# # #             # This would be a custom implementation based on your specific needs
            
# # #             return losses
# # #         else:
# # #             # During inference
# # #             detections = self.model(images)
            
# # #             # Add attribute predictions
# # #             for i, detection in enumerate(detections):
# # #                 boxes = detection['boxes']
                
# # #                 # Extract box features and predict attributes
# # #                 box_features = self._extract_box_features(images[i], boxes)
# # #                 attribute_predictions = self.attribute_head(box_features)
                
# # #                 # Add attribute predictions to detections
# # #                 detection.update(attribute_predictions)
            
# # #             return detections
    
# # #     def _format_targets(self, targets):
# # #         # Convert dataset targets to format expected by Faster R-CNN
# # #         # This is a placeholder - you'll need to implement based on your data structure
# # #         formatted_targets = []
# # #         # ... implementation ...
# # #         return formatted_targets
    
# # #     def _extract_box_features(self, image, boxes):
# # #         # Extract features for each box
# # #         # This is a placeholder - you'll need to implement this
# # #         # ... implementation ...
# # #         return features

# # # # Function to create the model
# # # def create_model(config):
# # #     model = CLEVRObjectDetector(config)
# # #     return model

# # # neuro_symbolic_visual_reasoning/neural_module/models/detector.py
# # import torch
# # import torch.nn as nn
# # import torchvision
# # from torchvision.models.detection import FasterRCNN
# # from torchvision.models.detection.rpn import AnchorGenerator

# # class AttributeHead(nn.Module):
# #     """
# #     Custom head for attribute classification (color, shape, material, size)
# #     """
# #     def __init__(self, in_features, num_colors, num_shapes, num_materials, num_sizes):
# #         super().__init__()
# #         self.common_fc = nn.Sequential(
# #             nn.Linear(in_features, 512),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #         )
        
# #         # Separate prediction heads for each attribute
# #         self.color_fc = nn.Linear(512, num_colors)
# #         self.shape_fc = nn.Linear(512, num_shapes)
# #         self.material_fc = nn.Linear(512, num_materials)
# #         self.size_fc = nn.Linear(512, num_sizes)
    
# #     def forward(self, x):
# #         x = self.common_fc(x)
# #         return {
# #             'color': self.color_fc(x),
# #             'shape': self.shape_fc(x),
# #             'material': self.material_fc(x),
# #             'size': self.size_fc(x)
# #         }

# # class CLEVRObjectDetector(nn.Module):
# #     """
# #     Object detector for CLEVR dataset with attribute classification
# #     """
# #     def __init__(self, config):
# #         super().__init__()
# #         self.config = config
        
# #         # Load backbone
# #         if config.BACKBONE == "mobilenet_v2":
# #             backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# #             backbone_out = 1280
# #         elif config.BACKBONE == "resnet18":
# #             backbone = torchvision.models.resnet18(pretrained=True)
# #             backbone_out = 512
# #             # Remove the last layers
# #             backbone = nn.Sequential(*list(backbone.children())[:-2])
# #         elif config.BACKBONE == "efficientnet_b0":
# #             backbone = torchvision.models.efficientnet_b0(pretrained=True).features
# #             backbone_out = 1280
# #         else:
# #             raise ValueError(f"Unsupported backbone: {config.BACKBONE}")
        
# #         # Freeze early layers for transfer learning
# #         for param in list(backbone.parameters())[:-10]:
# #             param.requires_grad = False
            
# #         # Add the out_channels attribute to the backbone
# #         backbone.out_channels = backbone_out
        
# #         # RPN anchor generator
# #         anchor_generator = AnchorGenerator(
# #             sizes=((32, 64, 128, 256),),
# #             aspect_ratios=((0.5, 1.0, 2.0),)
# #         )
        
# #         # ROI pooler
# #         roi_pooler = torchvision.ops.MultiScaleRoIAlign(
# #             featmap_names=['0'],
# #             output_size=7,
# #             sampling_ratio=2
# #         )
        
# #         # Create the Faster R-CNN model
# #         self.model = FasterRCNN(
# #             backbone,
# #             num_classes=2,  # Background and object
# #             rpn_anchor_generator=anchor_generator,
# #             box_roi_pool=roi_pooler
# #         )
        
# #         # Replace the box predictor with our custom one
# #         in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
# #         # Create attribute head
# #         self.attribute_head = AttributeHead(
# #             in_features, 
# #             len(config.COLORS),
# #             len(config.SHAPES),
# #             len(config.MATERIALS),
# #             len(config.SIZES)
# #         )
        
# #         # Replace the original box predictor
# #         self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, 2)  # Background and object
# #         self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, 4)  # Box coordinates
    
# #     def forward(self, images, targets=None):
# #         if self.training and targets is not None:
# #             # During training
# #             loss_dict = self.model(images, targets)
            
# #             # Extract box features for attribute classification
# #             # This would be customized in a full implementation
            
# #             return loss_dict
# #         else:
# #             # During inference
# #             detections = self.model(images)
            
# #             # Add attribute predictions
# #             # This would be customized in a full implementation
            
# #             return detections
    
# #     def _format_targets(self, targets):
# #         # Convert dataset targets to format expected by Faster R-CNN
# #         # This is a placeholder - you'll need to implement based on your data structure
# #         formatted_targets = []
# #         # ... implementation ...
# #         return formatted_targets
    
# #     def _extract_box_features(self, image, boxes):
# #         # Extract features for each box
# #         # This is a placeholder - you'll need to implement this
# #         # ... implementation ...
# #         return torch.rand(boxes.size(0), self.config.FEATURE_DIM)  # Placeholder

# # # Function to create the model
# # def create_model(config):
# #     model = CLEVRObjectDetector(config)
# #     return model

# # neuro_symbolic_visual_reasoning/neural_module/models/detector.py
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator

# class CLEVRObjectDetector(nn.Module):
#     """
#     Object detector for CLEVR dataset with attribute classification
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         # Load backbone
#         if config.BACKBONE == "mobilenet_v2":
#             backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#             backbone_out = 1280
#         elif config.BACKBONE == "resnet18":
#             backbone = torchvision.models.resnet18(pretrained=True)
#             backbone_out = 512
#             # Remove the last layers
#             backbone = nn.Sequential(*list(backbone.children())[:-2])
#         elif config.BACKBONE == "efficientnet_b0":
#             backbone = torchvision.models.efficientnet_b0(pretrained=True).features
#             backbone_out = 1280
#         else:
#             raise ValueError(f"Unsupported backbone: {config.BACKBONE}")
        
#         # Freeze early layers for transfer learning
#         for param in list(backbone.parameters())[:-10]:
#             param.requires_grad = False
            
#         # Add the out_channels attribute to the backbone
#         backbone.out_channels = backbone_out
        
#         # RPN anchor generator
#         anchor_generator = AnchorGenerator(
#             sizes=((32, 64, 128, 256),),
#             aspect_ratios=((0.5, 1.0, 2.0),)
#         )
        
#         # ROI pooler
#         roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#             featmap_names=['0'],
#             output_size=7,
#             sampling_ratio=2
#         )
        
#         # For CLEVR objects, we'll use a single foreground class (1) + background (0)
#         # The number of classes must include the background class
#         num_classes = 2  # Background + CLEVR object
        
#         # Create the Faster R-CNN model
#         self.model = FasterRCNN(
#             backbone,
#             num_classes=num_classes,
#             rpn_anchor_generator=anchor_generator,
#             box_roi_pool=roi_pooler,
#             min_size=320,  # Input image minimum size
#             max_size=480   # Input image maximum size
#         )
    
#     def forward(self, images, targets=None):
#         return self.model(images, targets)

# # Function to create the model
# def create_model(config):
#     model = CLEVRObjectDetector(config)
#     return model


import torch
import torch.nn as nn
from ultralytics import YOLO
import os

class CLEVRObjectDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize model
        if hasattr(config, 'MODEL_PATH') and config.MODEL_PATH and os.path.exists(config.MODEL_PATH):
            # Load from a previously trained model if path is specified
            print(f"Loading existing model from {config.MODEL_PATH}")
            self.model = YOLO(config.MODEL_PATH)
        else:
            # Start with a pretrained YOLOv8 Large model
            print(f"Creating new model from {config.MODEL}")
            self.model = YOLO(config.MODEL)
        
        # Set model parameters based on config
        self.conf_threshold = config.CONF_THRESHOLD
        self.device = config.DEVICE
    
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            # Training mode with targets
            results = self.model.train(data=images, targets=targets)
            return results
        else:
            # Inference mode
            results = self.model.predict(
                source=images,
                conf=self.conf_threshold,
                iou=self.config.NMS_THRESHOLD,
                device=self.device,
                verbose=False
            )
            return self._format_results(results)
    
    def _format_results(self, results):
        """Convert YOLO results to match expected format for symbolic reasoning"""
        if not results or len(results) == 0:
            return {
                'boxes': torch.zeros((0, 4), device=self.device),
                'scores': torch.zeros(0, device=self.device),
                'labels': torch.zeros(0, dtype=torch.int64, device=self.device)
            }
        
        result = results[0]  # Get first image results
        
        # Extract boxes, scores, and labels
        boxes = result.boxes.xyxy  # Get boxes in (x1, y1, x2, y2) format
        scores = result.boxes.conf
        labels = result.boxes.cls
        
        # Create detections dictionary
        detections = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
        
        return detections

    def train_model(self, data_yaml, epochs=100, batch_size=16, imgsz=640):
        """Train the YOLO model using Ultralytics API with settings optimized for CLEVR"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=20,       # Longer early stopping patience
            device=0 if torch.cuda.is_available() else 'cpu',
            project=self.config.MODEL_SAVE_DIR,
            name='yolo_clevr',
            save=True,
            lr0=0.001,        # Learning rate
            lrf=0.01,         # Final learning rate fraction
            warmup_epochs=5.0, # Warm-up epochs
            cos_lr=True,      # Use cosine learning rate scheduler
            augment=True,     # Enable data augmentation
            mosaic=1.0,       # Enable mosaic augmentation
            mixup=0.1,        # Enable mixup augmentation
            degrees=10.0,     # Rotation augmentation
            scale=0.25,       # Scale augmentation
            shear=5.0,        # Shear augmentation
            fliplr=0.5,       # Horizontal flip augmentation
            hsv_h=0.015,      # HSV hue augmentation
            hsv_s=0.7,        # HSV saturation augmentation
            hsv_v=0.4,        # HSV value augmentation
            cache=True,       # Cache images for faster training
            overlap_mask=True,# Better mask handling
            single_cls=True   # Enforce single class detection
        )
        return results

def create_model(config):
    model = CLEVRObjectDetector(config)
    return model
