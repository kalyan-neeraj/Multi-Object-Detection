from flask import Blueprint, request, jsonify
import torch
from PIL import Image
import io

# Create a Blueprint for the routes
bp = Blueprint('api', __name__)

# Load YOLOv5 model (make sure yolov5 is installed and weights are available)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Inference
    results = model(img)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return jsonify(detections)