from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import os
import sys

# --- New logic: Add parent directories to path for imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# --- New logic: Import from symbolic/neural modules ---
try:
    from symbolic_module.query_parser import QueryParser
    from symbolic_module.prolog_engine import PrologReasoner
    from symbolic_module.fact_converter import FactConverter
    from neural_module.inference import detect_objects, prepare_image
    from neural_module.config import Config
    from integration.pipeline import NeuroSymbolicPipeline
except ImportError:
    # If modules not found, fallback to mock
    QueryParser = PrologReasoner = FactConverter = None
    detect_objects = prepare_image = Config = None
    NeuroSymbolicPipeline = None

# --- Local utilities ---
from yolo_utils import yolo_object_detection, mock_object_detection, MODELS_DIR, DEFAULT_MODEL, DATA_DIR
from history_utils import query_history
from query_utils import mock_query_processing

app = Flask(__name__)
CORS(app)

# --- Data directory setup ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- New logic: Initialize pipeline ---
pipeline = None
try:
    print("Initializing NeuroSymbolic Pipeline...")
    pipeline = NeuroSymbolicPipeline()
    print("Successfully initialized NeuroSymbolic Pipeline")
except Exception as e:
    print(f"Error initializing NeuroSymbolic Pipeline: {str(e)}")
    print("Will fall back to mock implementation")

# =========================
# /predict endpoint
# =========================

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    image_path = os.path.join(DATA_DIR, "last_uploaded.jpg")
    img.save(image_path)

    # --- Read confidence threshold from request ---
    confidence_threshold = 0.02
    if request.form.get('confidence_threshold'):
        confidence_threshold = float(request.form.get('confidence_threshold'))
    elif request.json and request.json.get('confidence_threshold'):
        confidence_threshold = float(request.json.get('confidence_threshold'))

    # --- Use NeuroSymbolic pipeline if available ---
    if pipeline is not None:
        try:
            print(f"Processing image with NeuroSymbolic Pipeline (threshold={confidence_threshold})...")
            results = pipeline.process_image(image_path, visualize=False, threshold=confidence_threshold)
            objects = []
            if 'detections' in results and results['detections'] is not None:
                detections = results['detections']
                if 'boxes' in detections and len(detections['boxes']) > 0:
                    boxes = detections['boxes'].cpu().numpy()
                    scores = detections['scores'].cpu().numpy()
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        if score >= confidence_threshold:
                            x, y, x2, y2 = map(int, box)
                            width, height = x2 - x, y2 - y
                            # Extract attributes
                            color = "unknown"
                            shape = "object"
                            material = "unknown"
                            size = "unknown"
                            try:
                                if 'colors' in detections and i < len(detections['colors']):
                                    color_idx = detections['colors'][i].item()
                                    config = Config()
                                    color = config.COLORS[color_idx]
                                if 'shapes' in detections and i < len(detections['shapes']):
                                    shape_idx = detections['shapes'][i].item()
                                    config = Config()
                                    shape = config.SHAPES[shape_idx]
                                if 'materials' in detections and i < len(detections['materials']):
                                    material_idx = detections['materials'][i].item()
                                    config = Config()
                                    material = config.MATERIALS[material_idx]
                                if 'sizes' in detections and i < len(detections['sizes']):
                                    size_idx = detections['sizes'][i].item()
                                    config = Config()
                                    size = config.SIZES[size_idx]
                                else:
                                    area = width * height
                                    total_area = results['image_size'][0] * results['image_size'][1]
                                    size = "large" if area > total_area / 16 else "small"
                            except Exception as e:
                                print(f"Error extracting attributes: {str(e)}")
                            objects.append({
                                "id": f"obj{i+1}",
                                "label": shape.capitalize(),
                                "confidence": float(score),
                                "boundingBox": {"x": x, "y": y, "width": width, "height": height},
                                "attributes": {
                                    "color": color,
                                    "shape": shape,
                                    "size": size,
                                    "material": material
                                }
                            })
            print(f"Found {len(objects)} objects in the image")
            print("Objects:", objects)
            return jsonify({
                'objects': objects,
                'processingTime': results.get('processing_time', 0)
            })
        except Exception as e:
            import traceback
            print(f"Error in NeuroSymbolic pipeline: {str(e)}")
            traceback.print_exc()
            print("Falling back to mock implementation")
            # Use mock objects if pipeline fails
            objects = [
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
                }
            ]
            return jsonify({'objects': objects})

    # --- Original YOLO/mock fallback logic ---
    # Uncomment below if you want to use YOLO/mock as fallback or main
    # model_name = request.args.get('model', DEFAULT_MODEL)
    # try:
    #     detected_objects = yolo_object_detection(img, model_name=model_name)
    # except Exception:
    #     detected_objects = mock_object_detection(img)
    # return jsonify({'objects': detected_objects})

# =========================
# /query endpoint
# =========================

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query')
    detected_objects = data.get('objects', [])

    # --- New logic: Use symbolic pipeline if available ---
    if pipeline is not None:
        try:
            print(f"Processing query with NeuroSymbolic Pipeline: '{query_text}'")
            answer_result = pipeline.answer_question(query_text)
            # Format the answer
            if isinstance(answer_result['answer'], (int, bool)):
                if answer_result['parsed_query']['type'] == 'count':
                    answer = f"There {'is' if answer_result['answer'] == 1 else 'are'} {answer_result['answer']} {query_text.split('how many')[-1].strip()}."
                elif answer_result['parsed_query']['type'] == 'exist':
                    answer = f"{'Yes' if answer_result['answer'] else 'No'}, {query_text[:-1].lower()}."
                else:
                    answer = str(answer_result['answer'])
            elif isinstance(answer_result['answer'], list):
                if len(answer_result['answer']) == 1:
                    answer = f"The {answer_result['parsed_query']['params'].get('attribute', 'property')} is {answer_result['answer'][0]}."
                elif len(answer_result['answer']) > 1:
                    answer = f"The {answer_result['parsed_query']['params'].get('attribute', 'properties')} are: {', '.join(answer_result['answer'])}."
                else:
                    answer = "I couldn't determine the answer from the image."
            else:
                answer = str(answer_result['answer'])
            result = {
                'query': query_text,
                'answer': answer,
                'reasoning': answer_result.get('parsed_query', {}),
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }
            query_history.insert(0, result)
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"Error in symbolic reasoning: {str(e)}")
            traceback.print_exc()
            print("Falling back to mock implementation")

    # --- Original mock fallback logic ---
    print("Using mock query processing")
    answer = mock_query_processing(query_text, detected_objects)
    result = {
        'query': query_text,
        'answer': answer,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    query_history.insert(0, result)
    return jsonify(result)

# =========================
# /explain endpoint (new)
# =========================

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    query_text = data.get('query')
    if pipeline is not None:
        try:
            print(f"Generating explanation for query: '{query_text}'")
            explanation = pipeline.explain_reasoning(query_text)
            return jsonify(explanation)
        except Exception as e:
            import traceback
            print(f"Error generating explanation: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Symbolic reasoning pipeline not available'}), 500

# =========================
# History endpoints (shared)
# =========================

@app.route('/history', methods=['GET'])
def history():
    return jsonify(query_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    query_history.clear()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)