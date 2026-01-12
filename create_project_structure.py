# create_project_structure.py
import os

def create_directory_structure():
    # Main project directory
    base_dir = "neuro_symbolic_visual_reasoning"
    
    # Define the directory structure
    directories = [
        # Data directories
        "data/raw",
        "data/raw/images",
        "data/raw/scenes",
        "data/processed",
        "data/processed/images",
        "data/processed/scenes",
        "data/processed/questions",
        "data/preprocessing",
        
        # Neural module
        "neural_module/models",
        "neural_module/training",
        "neural_module/utils",
        
        # Symbolic module
        "symbolic_module",
        
        # Integration
        "integration",
        
        # Evaluation
        "evaluation",
        "evaluation/results",
        "evaluation/results/neural_only",
        "evaluation/results/neuro_symbolic",
        "evaluation/results/comparison",
        
        # Web interface
        "web_interface",
        "web_interface/static",
        "web_interface/templates",
        "web_interface/utils",
        
        # Notebooks
        "notebooks",
        
        # Scripts
        "scripts",
        
        # Tests
        "tests",
        "tests/test_data",
        
        # Documentation
        "docs"
    ]
    
    # Create each directory
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    # Create basic files
    files = {
        "requirements.txt": "# Project dependencies\nnumpy>=1.19.0\ntorch>=1.8.0\ntorchvision>=0.9.0\nmatplotlib>=3.3.0\npillow>=8.0.0\ntqdm>=4.50.0\nscikit-learn>=0.24.0\nopencv-python>=4.5.0\npyswip>=0.2.10\npyyaml>=5.4.0\nflask>=2.0.0\nstreamlit>=1.0.0\nultralytics>=8.0.0",
        "README.md": "# Neuro-Symbolic Visual Reasoning for Scene Understanding\n\nThis project implements a system that combines neural perception with symbolic reasoning for answering complex queries about visual scenes.",
        ".gitignore": "# Python\n__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nenv/\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\n*.egg-info/\n.installed.cfg\n*.egg\n\n# Jupyter Notebook\n.ipynb_checkpoints\n\n# Virtual Environment\nvenv/\nenv/\n\n# IDE files\n.idea/\n.vscode/\n*.swp\n*.swo\n\n# Project specific\ndata/raw/\ndata/processed/\n*.pt\n*.pth\n*.pkl\n"
    }
    
    for file_path, content in files.items():
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"Created file: {full_path}")

if __name__ == "__main__":
    create_directory_structure()
    print("Project structure created successfully!")