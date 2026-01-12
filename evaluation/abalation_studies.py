# test_complex_reasoning.py
import os
import sys
import argparse
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integration.pipeline import NeuroSymbolicPipeline
from neural_module.config import Config

def run_complex_visual_chain_test(model_path, image_dir, output_dir=None):
    """
    Run the Visual Chain test to evaluate complex spatial reasoning capabilities
    
    Args:
        model_path: Path to the trained model
        image_dir: Directory containing test images
        output_dir: Directory to save results (optional)
    """
    # Initialize the pipeline
    pipeline = NeuroSymbolicPipeline()
    try:
        pipeline.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Define complex test scenarios
    # Each scenario has a name, description, and a set of questions of increasing complexity
    test_scenarios = [
        {
            "name": "multi_hop_spatial_reasoning",
            "description": "Tests the ability to reason through multiple spatial relationships",
            "questions": [
                # 1-hop reasoning (direct spatial relationship)
                "What is to the left of the red sphere?",
                "How many objects are behind the blue cube?",
                
                # 2-hop reasoning (relation of relation)
                "What is the color of the object that is to the right of the blue cube?",
                "How many objects are to the left of the object behind the red sphere?",
                
                # 3-hop reasoning (relation of relation of relation)
                "What is the shape of the object that is to the right of the object that is behind the largest object?",
                "Is there an object in front of the object to the left of the blue object?",
                
                # 4-hop reasoning (very complex chain)
                "What is the color of the object that is to the right of the small object that is behind the object in front of the leftmost object?"
            ]
        },
        {
            "name": "multi_condition_filtering",
            "description": "Tests the ability to filter objects by multiple attribute and spatial conditions",
            "questions": [
                # Basic attribute conjunction
                "How many red objects are there?",
                "Are there any metal cubes?",
                
                # Attribute + spatial condition
                "How many blue objects are to the left of the red sphere?",
                "Is there a metal object behind the largest object?",
                
                # Multiple spatial conditions
                "How many objects are both to the left of the blue sphere and behind the red cube?",
                "Is there an object that is to the right of a metal object and in front of a small object?",
                
                # Complex filtering with negation
                "How many objects are not blue and not to the left of the largest object?",
                "What is the color of the object that is not a cube and not behind any sphere?"
            ]
        },
        {
            "name": "comparative_reasoning",
            "description": "Tests the ability to make comparisons between objects",
            "questions": [
                # Basic comparisons
                "Which object is larger: the red cube or the blue sphere?",
                "What color is the smallest object in the scene?",
                
                # Spatial comparisons
                "Which object is furthest to the right?",
                "What is the color of the object closest to the center?",
                
                # Complex comparisons
                "How many objects are smaller than the red cube?",
                "Is the blue sphere closer to the camera than the red cube?",
                
                # Superlatives with constraints
                "What is the color of the largest metal object?",
                "What shape is the rightmost object that is not blue?"
            ]
        },
        {
            "name": "between_reasoning",
            "description": "Tests the ability to reason about objects between other objects",
            "questions": [
                # Basic between questions
                "Is there anything between the red cube and the blue sphere?",
                "How many objects are between the leftmost and rightmost objects?",
                
                # Attribute filtering with between
                "How many red objects are between the blue cube and the green sphere?",
                "What is the material of the object between the largest and smallest objects?",
                
                # Complex between reasoning
                "Is there a row of at least three objects where the middle object is red?",
                "What is the shape of the object that is between an object that is blue and an object that is metal?"
            ]
        },
        {
            "name": "counterfactual_reasoning",
            "description": "Tests the ability to reason about hypothetical scenarios",
            "questions": [
                # Basic counterfactuals
                "If we removed the red cube, what would be the leftmost object?",
                "If the blue sphere were not there, how many objects would be visible?",
                
                # Spatial counterfactuals
                "If we removed all objects in front of the red cube, would the green sphere be visible?",
                "If the smallest object were moved to the right of the largest object, how many objects would be to its left?",
                
                # Complex counterfactuals
                "If all red objects were removed, what color would the largest remaining object be?",
                "If we swapped the positions of the red cube and blue sphere, which object would be closest to the green cylinder?"
            ]
        }
    ]
    
    # Get list of test images
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Results container
    results = {
        "model_path": model_path,
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": []
    }
    
    # Process each test scenario
    for scenario in test_scenarios:
        print(f"\nRunning scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        scenario_results = {
            "name": scenario['name'],
            "description": scenario['description'],
            "images": []
        }
        
        # Use a subset of images for each scenario
        # For demo purposes, limit to 3 images per scenario
        scenario_images = image_files[:3]
        
        for image_file in tqdm(scenario_images, desc="Processing images"):
            image_path = os.path.join(image_dir, image_file)
            
            # Process the image
            try:
                pipeline.process_image(image_path, visualize=False)
                
                image_result = {
                    "image_file": image_file,
                    "questions": []
                }
                
                # Ask each question
                for question in scenario['questions']:
                    start_time = time.time()
                    answer = pipeline.answer_question(question)
                    processing_time = time.time() - start_time
                    
                    question_result = {
                        "question": question,
                        "answer": answer['answer'],
                        "processing_time": processing_time
                    }
                    
                    image_result["questions"].append(question_result)
                    
                    # Optional: display progress
                    print(f"  Q: {question}")
                    print(f"  A: {answer['answer']}")
                    print(f"  Time: {processing_time:.3f}s")
                    print()
                
                scenario_results["images"].append(image_result)
                
                # Try to visualize a complex example from this scenario
                # Choose the most complex question (usually the last one)
                complex_question = scenario['questions'][-1]
                pipeline.run_multi_step_chain_test(image_path, complex_question)
                
            except Exception as e:
                print(f"Error processing image {image_file}: {str(e)}")
        
        results["scenarios"].append(scenario_results)
    
    # Calculate aggregate statistics
    success_rates = []
    processing_times = []
    
    for scenario in results["scenarios"]:
        scenario_success = 0
        scenario_total = 0
        scenario_times = []
        
        for image in scenario["images"]:
            for question in image["questions"]:
                scenario_total += 1
                if question["answer"] is not None:
                    scenario_success += 1
                    scenario_times.append(question["processing_time"])
        
        success_rate = scenario_success / max(1, scenario_total)
        avg_time = sum(scenario_times) / max(1, len(scenario_times))
        
        success_rates.append(success_rate)
        processing_times.append(avg_time)
        
        scenario["statistics"] = {
            "success_rate": success_rate,
            "average_processing_time": avg_time,
            "questions_asked": scenario_total,
            "questions_answered": scenario_success
        }
    
    results["overall_statistics"] = {
        "average_success_rate": sum(success_rates) / len(success_rates),
        "average_processing_time": sum(processing_times) / len(processing_times),
        "total_scenarios": len(results["scenarios"])
    }
    
    # Save results if output directory specified
    if output_dir:
        result_file = os.path.join(output_dir, "complex_reasoning_results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_file}")
    
    # Create and save visualizations of the results
    visualize_results(results, output_dir)
    
    return results

def visualize_results(results, output_dir=None):
    """Visualize test results with plots"""
    # Extract scenario names and success rates
    scenario_names = [s["name"] for s in results["scenarios"]]
    success_rates = [s["statistics"]["success_rate"] for s in results["scenarios"]]
    proc_times = [s["statistics"]["average_processing_time"] for s in results["scenarios"]]
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenario_names, success_rates, color='skyblue')
    plt.axhline(y=results["overall_statistics"]["average_success_rate"], 
                color='red', linestyle='--', label='Average')
    plt.xlabel('Scenario')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Reasoning Scenario')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'success_rates.png'))
    
    plt.show()
    
    # Plot processing times
    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenario_names, proc_times, color='lightgreen')
    plt.axhline(y=results["overall_statistics"]["average_processing_time"], 
                color='red', linestyle='--', label='Average')
    plt.xlabel('Scenario')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Average Processing Time by Reasoning Scenario')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'processing_times.png'))
    
    plt.show()
    
    # Create a heatmap of success rates by complexity
    # Average results across all images for each scenario and question complexity
    complexity_data = []
    
    for scenario in results["scenarios"]:
        # Calculate average success rate for each question position (complexity level)
        question_success = np.zeros(len(scenario["images"][0]["questions"]))
        question_count = np.zeros(len(scenario["images"][0]["questions"]))
        
        for image in scenario["images"]:
            for i, question in enumerate(image["questions"]):
                question_count[i] += 1
                if question["answer"] is not None:
                    question_success[i] += 1
        
        # Calculate success rates, avoiding division by zero
        success_rates = np.divide(question_success, question_count, 
                                  out=np.zeros_like(question_success), 
                                  where=question_count!=0)
        
        complexity_data.append(success_rates)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(complexity_data, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Success Rate')
    
    plt.xlabel('Question Complexity (increasing →)')
    plt.ylabel('Scenario')
    plt.title('Success Rate by Scenario and Question Complexity')
    
    plt.yticks(range(len(scenario_names)), scenario_names)
    plt.xticks(range(len(complexity_data[0])), 
               [f'Level {i+1}' for i in range(len(complexity_data[0]))])
    
    # Add text annotations
    for i in range(len(complexity_data)):
        for j in range(len(complexity_data[i])):
            plt.text(j, i, f'{complexity_data[i][j]:.2f}', 
                    ha='center', va='center', 
                    color='black' if complexity_data[i][j] > 0.5 else 'white')
    
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'complexity_heatmap.png'))
    
    plt.show()
    
    return

def run_ablation_study(model_path, image_dir, output_dir=None):
    """
    Run ablation study comparing symbolic reasoning vs. neural-only baseline
    
    Args:
        model_path: Path to the trained model
        image_dir: Directory containing test images
        output_dir: Directory to save results (optional)
    """
    # Initialize the neuro-symbolic pipeline
    neuro_symbolic_pipeline = NeuroSymbolicPipeline()
    try:
        neuro_symbolic_pipeline.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # For neural-only baseline, we'll use the same neural model but bypass the symbolic reasoning
    # Instead of converting to facts, we'll use a simple heuristic approach
    
    # Define test questions that vary in complexity
    test_questions = [
        # Simple questions (should work with both approaches)
        "How many objects are there in the scene?",
        "Is there a red object?",
        "What is the largest object?",
        
        # Medium questions (may work with neural heuristics)
        "How many objects are to the left of the red cube?",
        "What color is the object closest to the camera?",
        "Is there an object behind the blue sphere?",
        
        # Complex questions (should require symbolic reasoning)
        "What is the color of the object that is to the right of the small cube and behind the red sphere?",
        "How many objects are both to the left of a blue object and behind a small object?",
        "Is there a row of objects with the same color?"
    ]
    
    # Get list of test images
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    test_images = image_files[:5]  # Use a small subset for the ablation study
    
    # Results containers
    neuro_symbolic_results = []
    neural_only_results = []
    
    # Run tests with both approaches
    for image_file in test_images:
        image_path = os.path.join(image_dir, image_file)
        
        # Process with neuro-symbolic pipeline
        print(f"\nProcessing {image_file} with neuro-symbolic pipeline")
        try:
            neuro_symbolic_pipeline.process_image(image_path, visualize=False)
            
            for question in test_questions:
                start_time = time.time()
                answer = neuro_symbolic_pipeline.answer_question(question)
                processing_time = time.time() - start_time
                
                result = {
                    "image": image_file,
                    "question": question,
                    "answer": answer['answer'],
                    "processing_time": processing_time
                }
                
                neuro_symbolic_results.append(result)
                print(f"  Q: {question}")
                print(f"  A: {answer['answer']}")
                print(f"  Time: {processing_time:.3f}s")
        
        except Exception as e:
            print(f"Error with neuro-symbolic pipeline: {str(e)}")
        
        # Process with neural-only approach (simplified)
        print(f"\nProcessing {image_file} with neural-only approach")
        try:
            # Re-use the same image processing but skip symbolic reasoning
            neuro_symbolic_pipeline.process_image(image_path, visualize=False)
            
            for question in test_questions:
                start_time = time.time()
                
                # Simple heuristic reasoning based on neural detections only
                answer = neural_only_reasoning(neuro_symbolic_pipeline.current_detections, question)
                processing_time = time.time() - start_time
                
                result = {
                    "image": image_file,
                    "question": question,
                    "answer": answer,
                    "processing_time": processing_time
                }
                
                neural_only_results.append(result)
                print(f"  Q: {question}")
                print(f"  A: {answer}")
                print(f"  Time: {processing_time:.3f}s")
        
        except Exception as e:
            print(f"Error with neural-only approach: {str(e)}")
    
    # Analyze and compare results
    compare_approaches(neuro_symbolic_results, neural_only_results, output_dir)
    
    return neuro_symbolic_results, neural_only_results

def neural_only_reasoning(detections, question):
    """
    Simple heuristic reasoning based only on neural detections (no symbolic reasoning)
    This is a simplified baseline for comparison
    
    Args:
        detections: Neural detection results
        question: Question string
        
    Returns:
        Answer based on simple heuristics
    """
    # Extract detections
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    
    # Only consider confident detections
    valid_indices = np.where(scores >= 0.5)[0]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    
    question = question.lower()
    
    # Simple counting questions
    if "how many" in question:
        return len(valid_indices)
    
    # Simple existence questions
    if "is there" in question:
        if len(valid_indices) > 0:
            # For more specific existence questions, try to match keywords
            if "red" in question and hasattr(detections, 'colors'):
                # Check if any object is red
                # This is a placeholder - would need to check the actual colors
                return True
            return True  # Default to True if we have any objects
        return False
    
    # Simple attribute questions
    if "what color" in question:
        # Just return a default answer since we don't have real attribute classification
        return "red"  # Placeholder
    
    if "what shape" in question:
        return "cube"  # Placeholder
    
    # For complex spatial questions, we'll give a default answer
    if "right of" in question or "left of" in question or "behind" in question:
        return "Cannot determine with neural-only approach"
    
    # Default response
    return "Unable to answer with neural-only approach"

def compare_approaches(neuro_symbolic_results, neural_only_results, output_dir=None):
    """
    Compare the performance of the neuro-symbolic vs. neural-only approaches
    
    Args:
        neuro_symbolic_results: Results from neuro-symbolic pipeline
        neural_only_results: Results from neural-only approach
        output_dir: Directory to save comparison results
    """
    # Group questions by complexity
    question_complexity = {
        "simple": ["How many objects are there in the scene?", 
                   "Is there a red object?", 
                   "What is the largest object?"],
        "medium": ["How many objects are to the left of the red cube?", 
                   "What color is the object closest to the camera?", 
                   "Is there an object behind the blue sphere?"],
        "complex": ["What is the color of the object that is to the right of the small cube and behind the red sphere?", 
                    "How many objects are both to the left of a blue object and behind a small object?", 
                    "Is there a row of objects with the same color?"]
    }
    
    # Calculate success rates for each complexity level
    results = {
        "neuro_symbolic": {complexity: [] for complexity in question_complexity},
        "neural_only": {complexity: [] for complexity in question_complexity},
    }
    
    # Group the results by question complexity
    for result in neuro_symbolic_results:
        for complexity, questions in question_complexity.items():
            if result["question"] in questions:
                results["neuro_symbolic"][complexity].append(result)
    
    for result in neural_only_results:
        for complexity, questions in question_complexity.items():
            if result["question"] in questions:
                results["neural_only"][complexity].append(result)
    
    # Calculate success rates and processing times
    comparison = {
        "neuro_symbolic": {
            "success_rate": {},
            "avg_processing_time": {}
        },
        "neural_only": {
            "success_rate": {},
            "avg_processing_time": {}
        }
    }
    
    for approach in ["neuro_symbolic", "neural_only"]:
        for complexity in question_complexity:
            result_list = results[approach][complexity]
            
            # Count non-null/non-default answers as success
            if approach == "neuro_symbolic":
                successes = sum(1 for r in result_list if r["answer"] is not None and 
                               r["answer"] != "Unable to answer")
            else:
                successes = sum(1 for r in result_list if r["answer"] is not None and 
                               "Unable to answer" not in str(r["answer"]) and 
                               "Cannot determine" not in str(r["answer"]))
            
            success_rate = successes / max(1, len(result_list))
            comparison[approach]["success_rate"][complexity] = success_rate
            
            # Calculate average processing time
            times = [r["processing_time"] for r in result_list]
            avg_time = sum(times) / max(1, len(times))
            comparison[approach]["avg_processing_time"][complexity] = avg_time
    
    # Output comparison results
    print("\n===== APPROACH COMPARISON =====")
    for complexity in question_complexity:
        print(f"\n{complexity.upper()} QUESTIONS:")
        print(f"  Neuro-symbolic success rate: {comparison['neuro_symbolic']['success_rate'][complexity]:.2f}")
        print(f"  Neural-only success rate: {comparison['neural_only']['success_rate'][complexity]:.2f}")
        print(f"  Neuro-symbolic avg time: {comparison['neuro_symbolic']['avg_processing_time'][complexity]:.3f}s")
        print(f"  Neural-only avg time: {comparison['neural_only']['avg_processing_time'][complexity]:.3f}s")
    
    # Visualize the comparison
    complexities = list(question_complexity.keys())
    
    # Plot success rates
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(complexities))
    
    neuro_symbolic_rates = [comparison["neuro_symbolic"]["success_rate"][c] for c in complexities]
    neural_only_rates = [comparison["neural_only"]["success_rate"][c] for c in complexities]
    
    plt.bar(x - width/2, neuro_symbolic_rates, width, label='Neuro-Symbolic')
    plt.bar(x + width/2, neural_only_rates, width, label='Neural-Only')
    
    plt.xlabel('Question Complexity')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Approach and Question Complexity')
    plt.xticks(x, [c.capitalize() for c in complexities])
    plt.ylim(0, 1.0)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(neuro_symbolic_rates):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    for i, v in enumerate(neural_only_rates):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'approach_comparison_success.png'))
    
    plt.show()
    
    # Plot processing times
    plt.figure(figsize=(10, 6))
    
    neuro_symbolic_times = [comparison["neuro_symbolic"]["avg_processing_time"][c] for c in complexities]
    neural_only_times = [comparison["neural_only"]["avg_processing_time"][c] for c in complexities]
    
    plt.bar(x - width/2, neuro_symbolic_times, width, label='Neuro-Symbolic')
    plt.bar(x + width/2, neural_only_times, width, label='Neural-Only')
    
    plt.xlabel('Question Complexity')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time by Approach and Question Complexity')
    plt.xticks(x, [c.capitalize() for c in complexities])
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(neuro_symbolic_times):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}s', ha='center', va='bottom')
    
    for i, v in enumerate(neural_only_times):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'approach_comparison_time.png'))
    
    plt.show()
    
    # Save detailed comparison data
    if output_dir:
        with open(os.path.join(output_dir, 'approach_comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
    
    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complex reasoning tests on the neuro-symbolic system")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--images", required=True, help="Directory containing test images")
    parser.add_argument("--output", default="results", help="Directory to save results")
    parser.add_argument("--test_type", choices=["chain", "ablation"], default="chain",
                        help="Type of test to run (chain=Visual Chain Test, ablation=Ablation Study)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.test_type == "chain":
        print("Running Visual Chain Test...")
        run_complex_visual_chain_test(args.model, args.images, args.output)
    else:
        print("Running Ablation Study...")
        run_ablation_study(args.model, args.images, args.output)
    
    print(f"Testing completed. Results saved to {args.output}")