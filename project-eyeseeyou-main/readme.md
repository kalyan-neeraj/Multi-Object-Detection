# EyeSeeYou: Neuro-Symbolic Visual Reasoning System


A computer vision project that combines neural networks and symbolic reasoning to understand and answer questions about visual scenes.

## Overview

This project implements a neuro-symbolic visual reasoning system that:

- Detects objects and their attributes (shape, color, size) from synthetic scenes
- Answers complex compositional queries about spatial relationships
- Combines neural perception with symbolic logic for improved reasoning

### Architecture

```text
Input Image → Neural Module (CNN) → Symbolic Logic Engine → Natural Language Answer
```

## Features

- Object detection and attribute classification
- Spatial relationship reasoning
- Complex query processing
- Interactive web interface
- Support for CLEVR dataset

## Dataset

We use the CLEVR dataset from Stanford:

- Subsampled to 5k-10k images for manageable training
- Complex compositional questions
- Rich attribute annotations

## Setup

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (NVIDIA RTX 3060 or equivalent)
- 12GB+ RAM

### Installation

```bash
git clone https://github.com/yourusername/project-eyeseeyou.git
cd project-eyeseeyou
pip install -r requirements.txt
```

## Usage

1. Download and prepare the dataset

```bash
python scripts/prepare_data.py --sample-size 5000
```

1. Train the model

```bash
python train.py --epochs 100
```

1. Run the web interface

```bash
python app.py
```

## Project Timeline

- Weeks 1-2: Data Preparation
- Weeks 3-4: Neural Module Development
- Week 5: Symbolic Reasoning Integration
- Week 6: Pipeline Integration & Testing
- Week 7: Ablation Studies
- Week 8: Web Interface & Final Evaluation

## Evaluation Metrics

- Object Detection: mAP
- Question Answering Accuracy
- System Latency (<1s per query)
- Comparative Analysis with Baselines

## License

[MIT License](LICENSE)

## Contributors

- Nikhil Singh
- Aditya Singh
