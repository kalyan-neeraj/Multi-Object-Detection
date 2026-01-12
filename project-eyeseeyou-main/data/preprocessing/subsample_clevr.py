# subsample_clevr.py
import os
import json
import random
import shutil
from tqdm import tqdm

def organize_and_subsample_clevr(
    source_dir,
    destination_dir,
    num_samples=7500,  # Average between 5k-10k as mentioned in project plan
    random_seed=42
):
    """
    Organize CLEVR dataset and subsample to a manageable size (5k-10k as per project plan)
    
    Args:
        source_dir: Path to CLEVR_v1.0 directory
        destination_dir: Path to your project's data directory
        num_samples: Number of images to sample (default: 7500)
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create raw data directory
    raw_dir = os.path.join(destination_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Create processed (subsampled) data directory
    processed_dir = os.path.join(destination_dir, "processed")
    processed_images_dir = os.path.join(processed_dir, "images")
    processed_scenes_dir = os.path.join(processed_dir, "scenes")
    processed_questions_dir = os.path.join(processed_dir, "questions")
    
    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(processed_scenes_dir, exist_ok=True)
    os.makedirs(processed_questions_dir, exist_ok=True)
    
    # Load train scenes to get image filenames
    train_scenes_path = os.path.join(source_dir, "scenes", "CLEVR_train_scenes.json")
    
    print("Loading train scenes...")
    with open(train_scenes_path, 'r') as f:
        train_scenes = json.load(f)
    
    # Get all training image filenames
    all_train_images = [scene["image_filename"] for scene in train_scenes["scenes"]]
    print(f"Total training images: {len(all_train_images)}")
    print(f"Project plan requires: 5k-10k images (using {num_samples})")
    
    # Randomly select a subset of images
    selected_images = random.sample(all_train_images, min(num_samples, len(all_train_images)))
    print(f"Selected {len(selected_images)} images for subsampling")
    
    # Create a set of selected image filenames for faster lookup
    selected_images_set = set(selected_images)
    
    # Subsample and copy images
    source_images_dir = os.path.join(source_dir, "images", "train")
    print("Copying selected images...")
    for image_filename in tqdm(selected_images):
        source_path = os.path.join(source_images_dir, image_filename)
        dest_path = os.path.join(processed_images_dir, image_filename)
        shutil.copy2(source_path, dest_path)
    
    # Subsample scenes
    print("Subsampling scene annotations...")
    subsampled_scenes = {
        "info": train_scenes["info"],
        "scenes": [scene for scene in train_scenes["scenes"] 
                  if scene["image_filename"] in selected_images_set]
    }
    
    # Save subsampled scenes file
    subsampled_scenes_path = os.path.join(processed_scenes_dir, "CLEVR_subsampled_scenes.json")
    with open(subsampled_scenes_path, 'w') as f:
        json.dump(subsampled_scenes, f)
    
    # Load and subsample questions
    train_questions_path = os.path.join(source_dir, "questions", "CLEVR_train_questions.json")
    print("Loading train questions...")
    with open(train_questions_path, 'r') as f:
        train_questions = json.load(f)
    
    # Subsample questions
    print("Subsampling questions...")
    subsampled_questions = {
        "info": train_questions["info"],
        "questions": [q for q in train_questions["questions"] 
                     if q["image_filename"] in selected_images_set]
    }
    
    # Save subsampled questions file
    subsampled_questions_path = os.path.join(processed_questions_dir, "CLEVR_subsampled_questions.json")
    with open(subsampled_questions_path, 'w') as f:
        json.dump(subsampled_questions, f)
    
    # Create train/val split information
    print("Creating train/val split...")
    num_train = int(0.8 * len(selected_images))
    train_images = selected_images[:num_train]
    val_images = selected_images[num_train:]
    
    split_info = {
        "train": train_images,
        "val": val_images,
        "total_subsampled": len(selected_images),
        "original_total": len(all_train_images)
    }
    
    # Save split information
    split_info_path = os.path.join(processed_dir, "train_val_split.json")
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f)
    
    print(f"Subsampling complete! Created dataset with {len(selected_images)} images")
    print(f"Train set: {len(train_images)} images, Validation set: {len(val_images)} images")
    print(f"Scenes file saved to: {subsampled_scenes_path}")
    print(f"Questions file saved to: {subsampled_questions_path}")
    print(f"Split information saved to: {split_info_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize and subsample CLEVR dataset")
    parser.add_argument("--source", required=True, help="Path to CLEVR_v1.0 directory")
    parser.add_argument("--dest", required=True, help="Path to project data directory")
    parser.add_argument("--samples", type=int, default=7500, 
                        help="Number of images to sample (default: 7500, as per project plan range 5k-10k)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    organize_and_subsample_clevr(
        args.source,
        args.dest,
        args.samples,
        args.seed
    )