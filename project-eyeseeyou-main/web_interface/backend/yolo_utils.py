import os
from ultralytics import YOLO

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL = "yolov8n-seg.pt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

_loaded_models = {}

def get_yolo_model(model_name=DEFAULT_MODEL):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_name not in _loaded_models:
        _loaded_models[model_name] = YOLO(model_path)
    return _loaded_models[model_name]

def yolo_object_detection(img, model_name=DEFAULT_MODEL):
    model = get_yolo_model(model_name)
    results = model(img)
    objects = []
    for i, box in enumerate(results[0].boxes):
        label_idx = int(box.cls[0])
        label = results[0].names[label_idx]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        x, y, x2, y2 = map(int, xyxy)
        width, height = x2 - x, y2 - y
        attributes = {
            "color": "unknown",
            "shape": label,
            "size": "unknown",
            "material": "unknown"
        }
        mask = None
        if hasattr(results[0], "masks") and results[0].masks is not None:
            mask = results[0].masks.data[i].cpu().numpy().tolist()
        objects.append({
            "id": f"obj{i+1}",
            "label": label.capitalize(),
            "confidence": conf,
            "boundingBox": {"x": x, "y": y, "width": width, "height": height},
            "attributes": attributes,
            "mask": mask
        })
    return objects

def mock_object_detection(img):
    return [
        {
            "id": "obj1",
            "label": "Cube",
            "confidence": 0.92,
            "boundingBox": {"x": 150, "y": 100, "width": 100, "height": 100},
            "attributes": {"color": "red", "shape": "cube", "size": "medium", "material": "plastic"}
        },
        {
            "id": "obj2",
            "label": "Sphere",
            "confidence": 0.87,
            "boundingBox": {"x": 300, "y": 200, "width": 80, "height": 80},
            "attributes": {"color": "blue", "shape": "sphere", "size": "small", "material": "metal"}
        },
        {
            "id": "obj3",
            "label": "Cylinder",
            "confidence": 0.85,
            "boundingBox": {"x": 450, "y": 150, "width": 60, "height": 120},
            "attributes": {"color": "green", "shape": "cylinder", "size": "large", "material": "wood"}
        },
        {
            "id": "obj4",
            "label": "Pyramid",
            "confidence": 0.78,
            "boundingBox": {"x": 200, "y": 300, "width": 90, "height": 90},
            "attributes": {"color": "yellow", "shape": "pyramid", "size": "medium", "material": "plastic"}
        },
        {
            "id": "obj5",
            "label": "Cone",
            "confidence": 0.73,
            "boundingBox": {"x": 350, "y": 100, "width": 70, "height": 100},
            "attributes": {"color": "purple", "shape": "cone", "size": "small", "material": "rubber"}
        }
    ]
