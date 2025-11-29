# H-MTL: Hierarchical Severity-Aware Multi-Task Learning for Robot Fault Diagnosis

## 📋 Overview

This repository implements a **Hierarchical Severity-Aware Multi-Task Learning (H-MTL)** framework for progressive fault diagnosis in semiconductor transfer robot belt drives. The model jointly performs fault-type classification as the main task and severity-level estimation as the auxiliary task.

### Key Features
- 🔍 **Fault-type Classification**: Normal → Tension Reduction → Wear (3 classes)
- 📊 **Severity Estimation**: Light → Medium → Severe (3 levels per fault type)
- 🧠 **SPF Module**: Severity Pattern Fusion for degradation encoding
- 🔄 **IKR Module**: Iterative Knowledge Refinement (3 cycles)
- 📈 **EMD Loss**: Earth Mover's Distance for ordinal relationships
- 📉 **ACR Metric**: Adjacent Confusion Rate for ordinal accuracy

## 🏗️ Architecture

### Model Components

```
Input (Batch, 2, 780)
    ↓
┌─────────────────────┐
│   CNN Backbone      │ → Feature extraction (2→16→32→64→128)
└─────────────────────┘
    ↓
┌─────────────────────┐
│   SPF Module        │ → Severity Pattern Fusion
│  (Tension/Wear)     │   • Domain-specific features
└─────────────────────┘   • Severity embeddings
    ↓                     • Integrated knowledge
┌─────────────────────┐
│   IKR Module        │ → Iterative Refinement (K=3)
│  (3 Iterations)     │   • Bidirectional knowledge transfer
└─────────────────────┘   • Multi-head attention
    ↓                     • Weighted aggregation
Main Task  Sub Tasks
[Classifier] [Classifiers]
```

### Module Details

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| **Backbone** | 1D CNN feature extraction | (B, 2, 780) | (B, 128) |
| **SPF** | Severity-aware representations | (B, 128) | (B, 128) × 3 domains |
| **IKR** | Iterative knowledge exchange | (B, 128) × 3 | (B, 128) × 3 (refined) |
| **Classifiers** | Task predictions | (B, 128) | (B, 3) logits |

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd H-MTL-FaultDiagnosis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```

## 🚀 Quick Start

### 1. Data Preparation

```python
from src.utils.dataset import load_industrial_data

# Load semiconductor robot vibration dataset
train_loader, test_loader = load_industrial_data(
    data_dir='data/robot_dataset',
    batch_size=64,
    seq_len=780
)
```

### 2. Model Training

```python
from src.models.h_mtl_model import H_MTL_Model
from src.train import train_model

# Initialize model
model = H_MTL_Model(seq_len=780, hidden_dim=128, num_iterations=3)

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=300,
    learning_rate=1e-3,
    device='cuda'
)
```

### 3. Evaluation

```python
from src.evaluate import evaluate_model

# Evaluate on test set
results = evaluate_model(
    model=model,
    test_loader=test_loader,
    metrics=['accuracy', 'acr', 'confusion_matrix']
)

print(f"Main Task Accuracy: {results['main_accuracy']:.4f}")
print(f"Hierarchical Accuracy: {results['hierarchical_accuracy']:.4f}")
print(f"Adjacent Confusion Rate: {results['acr']:.2f}%")
```

## 📊 Experimental Results

### Industrial Semiconductor Robot Dataset
- **Train Samples**: 7,000 cycles (1,000 per class)
- **Test Samples**: 1,400 cycles (200 per class)
- **Sampling Rate**: 25.6 kHz
- **Signal Duration**: 7.8 seconds per cycle

#### Performance Metrics

| Metric | Main Task | Hierarchical (7-class) |
|--------|-----------|------------------------|
| **Accuracy** | 100.0% | 98.90% |
| **Precision** | 100.0% | 98.87% |
| **Recall** | 100.0% | 98.99% |
| **F1-Score** | 100.0% | 98.83% |
| **ACR** | - | 1.14% |

### CWRU Bearing Dataset Benchmarking

| Model | Main Acc | Sub Acc | Macro-F1 | ACR |
|-------|----------|---------|----------|-----|
| MSCNN-LSTM | 96.9% | 96.38% | 0.9632 | 8.94% |
| CNN-Transformer | 97.13% | 96.9% | 0.9687 | 8.62% |
| ResNet18 | 97.55% | 97.45% | 0.9742 | 7.45% |
| **H-MTL (Proposed)** | **97.89%** | **97.55%** | **0.9753** | **6.49%** |

## 🔧 Configuration

Edit `configs/default.yaml`:

```yaml
# Model Configuration
model:
  hidden_dim: 128
  num_iterations: 3
  input_seq_len: 780

# Training Configuration
train:
  batch_size: 64
  epochs: 300
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4

# Loss Weights
loss:
  lambda_task: 1.0
  lambda_struct: 0.7
  lambda_aux: 0.3

# Data Configuration
data:
  train_split: 0.8
  seq_len: 780
  num_classes: 3
  num_severity_levels: 3
```

## 📈 Key Findings

### Ablation Study

| Configuration | Main Acc | 7-class Acc | F1-Score | ACR |
|---------------|----------|------------|----------|-----|
| Base (CNN-MTL) | 98.86% | 95.88% | 0.9590 | 8.71% |
| + SPF | 99.64% | 97.71% | 0.9772 | 3.00% |
| + SPF + IKR | 100.00% | 98.57% | 0.9857 | 1.43% |
| + SPF + IKR + EMD | 100.00% | 98.90% | 0.9890 | 1.14% |

### Iteration Analysis

- **Iteration 1**: Foundation knowledge transfer
- **Iteration 2**: Refined cross-task learning  
- **Iteration 3**: Final knowledge distillation
- **Optimal K**: 3 iterations (convergence achieved)

## 🎯 Model Insights

### Severity Pattern Fusion (SPF)
- Encodes degradation progression from normal → light → medium → severe
- Generates severity embeddings for tension and wear domains
- Produces continuous severity scores for auxiliary guidance

### Iterative Knowledge Refinement (IKR)
- **Step 1**: Feature exchange between tasks (linear transformation)
- **Step 2**: Multi-head attention for nonlinear dependencies
- **Step 3**: Residual update with normalization
- **Aggregation**: Softmax-weighted combination of K iterations

### Loss Design
- **Task Loss**: Standard cross-entropy for classification
- **Structural Loss**: EMD-based ordinal loss (preserves severity ordering)
- **Auxiliary Loss**: MSE for continuous severity prediction

## 📁 Project Structure

```
H-MTL-FaultDiagnosis/
├── README.md
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── default.yaml
│   └── experiment_robot.yaml
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py          # CNN feature extractor
│   │   ├── spf_module.py         # Severity Pattern Fusion
│   │   ├── ikr_module.py         # Iterative Knowledge Refinement
│   │   └── h_mtl_model.py        # Main model
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Data loading
│   │   ├── metrics.py            # Loss & metrics
│   │   └── visualization.py      # Plotting utilities
│   │
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── inference.py              # Inference utilities
│
├── experiments/
│   ├── robot_experiment.py
│   └── cwru_experiment.py
│
└── figures/
    ├── architecture.png
    ├── results_confusion.png
    └── tsne_visualization.png
```

## 🔬 Reproducibility

### Experimental Setup
- **Hardware**: GPU (NVIDIA RTX 4070 Ti Super, 16GB VRAM)
- **Framework**: PyTorch 2.0+
- **Precision**: 32-bit floating point
- **Random Seed**: 42 (for reproducibility)


