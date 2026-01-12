# integration/pipeline.py
import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import neural module components
from neural_module.config import Config
from neural_module.inference import detect_objects, prepare_image

# Import symbolic module components
from symbolic_module.fact_converter import FactConverter
from symbolic_module.prolog_engine import PrologReasoner
from symbolic_module.query_parser import QueryParser

class NeuroSymbolicPipeline:
    """
    Pipeline that connects neural object detection with symbolic reasoning
    """
    def __init__(self):
        # Initialize configuration
        self.config = Config()
        print(f"Initialized Config with device: {self.config.DEVICE}")
        
        # Make sure model path is set
        if self.config.MODEL_PATH is None or not os.path.exists(self.config.MODEL_PATH):
            print(f"WARNING: Model path not found or not set: {self.config.MODEL_PATH}")
            # Set model path to the correct file name - best.pt
            self.config.MODEL_PATH = os.path.join(project_root, "models", "saved", "best.pt")
            print(f"Using model path: {self.config.MODEL_PATH}")
        
        # Initialize neural model - this part depends on your specific detector implementation
        self.model = self._load_model(self.config.MODEL_PATH)
        print(f"Model loaded from {self.config.MODEL_PATH}")
        
        # Initialize symbolic components
        self.fact_converter = FactConverter(self.config)
        self.prolog_engine = PrologReasoner()
        self.query_parser = QueryParser(self.config)
        
        # Store the current knowledge base and image
        self.current_knowledge_base = None
        self.current_image_path = None
        self.current_image_size = None
        
        print("NeuroSymbolic Pipeline initialized successfully.")
    
    def _load_model(self, model_path):
        """Load the neural object detection model"""
        try:
            # Import at runtime to handle potential import errors gracefully
            from ultralytics import YOLO
            
            # Check if model exists
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                model = YOLO(model_path)
                return model
            else:
                print(f"Model not found at {model_path}, checking for best.pt")
                # Try one more location before falling back to default
                alternate_path = os.path.join(os.path.dirname(model_path), "best.pt")
                if os.path.exists(alternate_path):
                    print(f"Found best.pt at {alternate_path}")
                    model = YOLO(alternate_path)
                    return model
                else:
                    print("Using default YOLO model")
                    model = YOLO("yolov8n.pt")  # Use a smaller default model
                    return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Using mock model instead")
            return MockModel()  # Implement a mock model class for fallback
    
    def process_image(self, image_path, visualize=False, threshold=0.01):
        """
        Process an image through the neural and symbolic pipeline
        
        Args:
            image_path: Path to the image file
            visualize: Whether to visualize the detections
            threshold: Confidence threshold for detections
        
        Returns:
            Dictionary containing detection results and processing time
        """
        start_time = time.time()
        
        print(f"Processing image: {image_path}")
        self.current_image_path = image_path
        
        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            self.current_image_size = image.size
            print(f"Image loaded with size: {self.current_image_size}")
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return {
                'error': f"Failed to load image: {str(e)}",
                'processing_time': time.time() - start_time
            }
        
        # Detect objects using the neural model
        try:
            detections = detect_objects(self.model, image_path, image_np, threshold)
            print(f"Detection complete. Found objects: {len(detections['boxes'])}")
        except Exception as e:
            print(f"Error detecting objects: {str(e)}")
            return {
                'error': f"Failed to detect objects: {str(e)}",
                'processing_time': time.time() - start_time
            }
        
        # Convert detections to symbolic knowledge base
        try:
            knowledge_base = self.fact_converter.generate_knowledge_base(
                detections, self.current_image_size[0], self.current_image_size[1], threshold
            )
            
            # Store the knowledge base for later access
            self.current_knowledge_base = knowledge_base
            
            # Load the knowledge base into the Prolog engine
            self.prolog_engine.load_knowledge_base(knowledge_base)
            
            print(f"Generated knowledge base with {knowledge_base.count('.')} facts and rules")
        except Exception as e:
            print(f"Error generating knowledge base: {str(e)}")
            return {
                'error': f"Failed to generate knowledge base: {str(e)}",
                'detections': detections,
                'image_size': self.current_image_size,
                'processing_time': time.time() - start_time
            }
        
        processing_time = time.time() - start_time
        
        return {
            'detections': detections,
            'image_size': self.current_image_size,
            'processing_time': processing_time
        }
    
    def answer_question(self, question):
        """
        Answer a question about the processed image
        
        Args:
            question: Natural language question about the image
        
        Returns:
            Dictionary containing the answer and reasoning steps
        """
        if self.current_knowledge_base is None:
            return {
                'answer': "No image has been processed. Please process an image first.",
                'parsed_query': {'type': 'unknown', 'params': {}}
            }
        
        print(f"Answering question: '{question}'")
        
        # Parse the question
        try:
            parsed_query = self.query_parser.parse_question(question)
            print(f"Parsed query: {parsed_query}")
        except Exception as e:
            print(f"Error parsing question: {str(e)}")
            return {
                'answer': f"Failed to parse question: {str(e)}",
                'parsed_query': {'type': 'unknown', 'params': {}}
            }
        
        # Answer the question using symbolic reasoning
        try:
            answer = self.prolog_engine.answer_question(
                parsed_query['type'],
                parsed_query['params']
            )
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return {
                'answer': f"Failed to answer question: {str(e)}",
                'parsed_query': parsed_query
            }
        
        return {
            'answer': answer,
            'parsed_query': parsed_query
        }
    
    def explain_reasoning(self, question):
        """
        Explain the reasoning process for answering a question
        
        Args:
            question: Natural language question about the image
        
        Returns:
            Dictionary containing the explanation steps
        """
        if self.current_knowledge_base is None:
            return {
                'error': "No image has been processed. Please process an image first.",
                'question': question,
                'query_type': 'unknown'
            }
        
        # Parse the question
        parsed_query = self.query_parser.parse_question(question)
        query_type = parsed_query['type']
        
        explanation = {
            'question': question,
            'query_type': query_type,
            'steps': []
        }
        
        if query_type == 'count':
            params = parsed_query['params']
            base_query = "object(X)"
            constraints = []
            
            # Add constraints for the main object
            for attr in ['color', 'shape', 'material', 'size']:
                if attr in params:
                    constraints.append(f"{attr}(X, {params[attr]})")
            
            # First step: find all objects
            step1 = {
                'description': "Finding all objects in the scene.",
                'count': self.prolog_engine.count_solutions("object(X)")
            }
            explanation['steps'].append(step1)
            
            # Second step: apply constraints (if any)
            if constraints:
                query = base_query + ", " + ", ".join(constraints)
                step2 = {
                    'description': f"Filtering objects with constraints: {', '.join(constraints)}",
                    'count': self.prolog_engine.count_solutions(query)
                }
                explanation['steps'].append(step2)
            
            # Final step: apply spatial relations (if any)
            if 'relation' in params and 'rel_object' in params:
                rel = params['relation']
                rel_constraints = []
                
                for attr in ['color', 'shape', 'material', 'size']:
                    rel_attr = f'rel_{attr}'
                    if rel_attr in params:
                        rel_constraints.append(f"{attr}(Y, {params[rel_attr]})")
                
                rel_query = "object(Y)"
                if rel_constraints:
                    rel_query += ", " + ", ".join(rel_constraints)
                
                # First find the related objects
                step3a = {
                    'description': f"Finding objects matching related object criteria: {', '.join(rel_constraints)}",
                    'count': self.prolog_engine.count_solutions(rel_query)
                }
                explanation['steps'].append(step3a)
                
                # Then find objects with the relation
                full_query = query + ", " + rel_query + f", {rel}(X, Y)"
                step3b = {
                    'description': f"Finding objects that are {rel.replace('_', ' ')} the related objects",
                    'count': self.prolog_engine.count_solutions(full_query),
                    'answer': self.prolog_engine.count_solutions(full_query)
                }
                explanation['steps'].append(step3b)
        
        elif query_type == 'exist':
            # Similar to count, but with exist check
            params = parsed_query['params']
            explanation['steps'].append({
                'description': "Checking if objects matching criteria exist",
                'answer': self.prolog_engine.answer_question('exist', params)
            })
        
        elif query_type == 'query_attribute':
            params = parsed_query['params']
            attr = params.get('attribute', 'color')
            
            step1 = {
                'description': f"Finding objects matching constraints",
                'count': self.prolog_engine.count_solutions("object(X)")
            }
            explanation['steps'].append(step1)
            
            # Build query based on constraints
            constraints = []
            for a in ['color', 'shape', 'material', 'size']:
                if a != attr and a in params:
                    constraints.append(f"{a}(X, {params[a]})")
            
            if constraints:
                query = "object(X), " + ", ".join(constraints)
                step2 = {
                    'description': f"Applying constraints: {', '.join(constraints)}",
                    'count': self.prolog_engine.count_solutions(query)
                }
                explanation['steps'].append(step2)
            
            # Apply attribute query
            query = "object(X)"
            if constraints:
                query += ", " + ", ".join(constraints)
            query += f", {attr}(X, Value)"
            
            results = self.prolog_engine.query(query)
            values = [r['Value'] for r in results] if results else []
            
            step3 = {
                'description': f"Determining the {attr} value",
                'answer': values[0] if values else None,
                'objects': []
            }
            explanation['steps'].append(step3)
        
        # Add more explanation logic for other query types
        
        return explanation

# Mock model class for fallback
class MockModel:
    """Simple mock model to use as fallback if real model loading fails"""
    def __init__(self):
        self.name = "MockModel"
        self.device = "cpu"
    
    def __call__(self, img, *args, **kwargs):
        # Return mock results
        return self.predict(img, *args, **kwargs)
        
    def predict(self, img, *args, **kwargs):
        # Return mock results
        class MockResults:
            def __init__(self):
                self.boxes = Boxes()
                self.name = "mock_results"
                
            def __getitem__(self, idx):
                return self
                
        class Boxes:
            def __init__(self):
                self.xyxy = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
                self.conf = torch.tensor([0.9, 0.8])
                self.cls = torch.tensor([0, 1])
                
        return [MockResults()]
    
    def _format_results(self, results):
        """Format mock results into detections dictionary"""
        try:
            # Try to get from boxes attribute
            boxes = results[0].boxes.xyxy
            scores = results[0].boxes.conf
            classes = results[0].boxes.cls
        except:
            # Fallback to direct creation
            boxes = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
            scores = torch.tensor([0.9, 0.8])
            classes = torch.tensor([0, 1])
            
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes
        }
# class MockModel:
#     """Simple mock model to use as fallback if real model loading fails"""
#     def __init__(self):
#         self.name = "MockModel"
    
#     def __call__(self, img, *args, **kwargs):
#         # Return mock results
#         class MockResults:
#             def __init__(self):
#                 self.boxes = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
#                 self.conf = torch.tensor([0.9, 0.8])
#                 self.cls = torch.tensor([0, 1])
        
#         return [MockResults()]