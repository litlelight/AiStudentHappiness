# PERMA-Guided Multi-Topology Graph Neural Networks for Cross-Cultural Student Well-being Prediction

## Overview

This repository contains the implementation of **PERMA-GNN-Transformer**, a novel deep learning framework that integrates positive psychology theory with graph neural networks for predicting student well-being across different cultural contexts. The model achieves state-of-the-art performance by combining PERMA (Positive Emotion, Engagement, Relationships, Meaning, and Accomplishment) theory-driven feature embedding, multi-topology graph neural networks, and Transformer-based encoding.

**Key Highlights:**
- Integrates PERMA positive psychology theory with deep learning architecture
- Multi-topology graph structure modeling (cosine similarity, Euclidean distance, learning styles, PERMA-weighted graphs)
- Cross-cultural adaptability validated on datasets from multiple countries
- Achieves 42.8% improvement over baseline methods on large-scale datasets
- Provides interpretable multi-dimensional well-being predictions

## Project Structure

```
├── data/
│   ├── preprocess.py          # Data preprocessing and feature extraction
│   ├── graph_construction.py  # Multi-topology graph construction
│   └── dataset_loader.py      # Custom dataset loaders
├── models/
│   ├── perma_gnn_transformer.py  # Main model architecture
│   ├── perma_embedding.py        # PERMA theory-driven feature embedding
│   ├── multi_topology_gnn.py     # Multi-topology graph neural network
│   ├── perma_transformer.py      # PERMA-aligned Transformer encoder
│   └── loss_functions.py         # Multi-task loss with consistency constraints
├── utils/
│   ├── metrics.py             # PERMA theory comprehensive evaluation metrics
│   ├── visualization.py       # Result visualization tools
│   └── config.py              # Configuration management
├── experiments/
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Model evaluation script
│   └── ablation_study.py      # Ablation study experiments
├── configs/
│   ├── default_config.yaml    # Default configuration file
│   └── hyperparameters.yaml   # Hyperparameter settings
├── notebooks/
│   ├── data_exploration.ipynb      # Data exploration and analysis
│   └── result_visualization.ipynb  # Results visualization
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Datasets

This study utilizes two publicly available datasets to validate cross-cultural adaptability:

### 1. Lifestyle and Wellbeing Data
- **Source**: Kaggle ([www.Authentic-Happiness.com](https://github.com/litlelight/AiStudentHappiness/blob/main/Dataset01.csv))
- **Size**: 12,757 valid samples
- **Features**: 23 dimensions including social time, exercise habits, sleep quality, work hours, etc.
- **Cultural Context**: Primarily Western cultural backgrounds
- **Access Date**: March 15, 2024
- **Ethics**: Public dataset with appropriate consent

### 2. International Student Mental Health Dataset
- **Source**: International university in Japan
- **Size**: 268 students (50% international, 50% domestic)
- **Features**: Mental health surveys, academic performance, social adaptation indicators
- **Cultural Context**: Multi-cultural environment (Asian and international students)
- **Access Date**: March 18, 2024
- **Ethics**: Anonymized with appropriate institutional approval

**Data Mapping to PERMA Theory:**
- Positive Emotion: Mood scores, satisfaction ratings
- Engagement: Study time, activity participation
- Relationships: Social time, peer interaction frequency
- Meaning: Academic goals, life purpose indicators
- Accomplishment: Academic performance, achievement milestones

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ GPU VRAM (NVIDIA RTX 4090 or equivalent recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/PERMA-GNN-Transformer.git
cd PERMA-GNN-Transformer
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Core Dependencies
- PyTorch 2.0.1+cu118
- PyTorch Geometric 2.3.1
- NumPy 1.24.3
- Pandas 2.0.2
- Scikit-learn 1.3.0
- Matplotlib 3.7.1
- Seaborn 0.12.2
- PyYAML 6.0
- tqdm 4.65.0

## Quick Start

### 1. Data Preprocessing
```bash
python data/preprocess.py --dataset lifestyle --output data/processed/
python data/preprocess.py --dataset international --output data/processed/
```

### 2. Graph Construction
```bash
python data/graph_construction.py --input data/processed/ --topologies all
```

### 3. Model Training
```bash
# Train on Lifestyle and Wellbeing Data
python experiments/train.py --config configs/default_config.yaml --dataset lifestyle

# Train on International Student Mental Health Dataset
python experiments/train.py --config configs/default_config.yaml --dataset international
```

### 4. Model Evaluation
```bash
python experiments/evaluate.py --checkpoint checkpoints/best_model.pth --dataset lifestyle
```

## Model Architecture

The PERMA-GNN-Transformer consists of four key components:

1. **Multi-source Feature Input & Graph Construction**
   - Constructs four topology graphs: cosine similarity, Euclidean distance, learning styles, PERMA-weighted
   - Captures diverse student relationship patterns

2. **PERMA Theory-Driven Feature Embedding**
   - Maps raw educational features to five PERMA dimensions
   - Cross-modal attention mechanism for dimension interaction modeling

3. **Multi-topology Graph Neural Network**
   - Parallel GCN and GAT layers for each topology
   - Graph-level attention fusion for adaptive topology weighting

4. **PERMA-Aligned Transformer Encoder**
   - Five-head attention mechanism aligned with PERMA dimensions
   - Multi-task learning framework with consistency constraints

## Hyperparameter Configuration

### Key Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2×10⁻⁴ | Initial learning rate |
| Batch Size | 32 | Training batch size |
| Hidden Dimension | 256 | Feature embedding dimension |
| GCN Layers | 3 | Number of GCN layers per topology |
| GCN Hidden Dim | 128 | Hidden dimension for GCN layers |
| GAT Heads | 8 | Number of attention heads in GAT |
| Transformer Heads | 5 | Number of attention heads (PERMA-aligned) |
| Dropout Rate | 0.3 | Dropout probability |
| Weight Decay | 1×10⁻⁵ | L2 regularization coefficient |
| Max Epochs | 200 | Maximum training epochs |
| Early Stopping | 20 | Patience for early stopping |

**Complete hyperparameter details are provided in Appendix A of the manuscript.**

### Optimizer Settings
- **Optimizer**: AdamW
- **β₁**: 0.9
- **β₂**: 0.999
- **ε**: 1×10⁻⁸
- **Learning Rate Scheduler**: Cosine Annealing

### Weight Initialization
- **Feature Embedding**: Xavier Uniform
- **GNN Layers**: He Normal
- **Transformer**: Xavier Uniform with scaling

## Evaluation Metrics

### Traditional Metrics
- **MAE (Mean Absolute Error)**: Average absolute deviation
- **RMSE (Root Mean Square Error)**: Root mean squared error
- **R² Score**: Coefficient of determination

### PERMA Theory Comprehensive Metrics
- **PDA (PERMA Dimension Accuracy)**: Prediction accuracy across five PERMA dimensions
- **PCI (PERMA Consistency Index)**: Theoretical consistency between overall and dimensional predictions
- **PCE (PERMA Comprehensive Effectiveness)**: Combined evaluation of accuracy and consistency

## Reproducing Results

### Main Comparative Experiments
```bash
# Run all baseline comparisons
python experiments/comparative_experiments.py --config configs/default_config.yaml
```

**Expected Results (Lifestyle Dataset):**
- MAE: 0.163
- RMSE: 0.206
- R²: 0.912
- PDA: 0.841
- PCI: 0.792
- PCE: 0.843

**Expected Results (International Student Dataset):**
- MAE: 0.147
- RMSE: 0.189
- R²: 0.927
- PDA: 0.823
- PCI: 0.784
- PCE: 0.835

### Ablation Study
```bash
python experiments/ablation_study.py --config configs/default_config.yaml
```

### Hyperparameter Sensitivity Analysis
```bash
python experiments/hyperparameter_tuning.py --param learning_rate --range 1e-5 1e-3
python experiments/hyperparameter_tuning.py --param batch_size --range 16 128
```


```

## Visualization

Generate result visualizations:
```bash
python utils/visualization.py --results results/predictions.csv --output figures/
```

Available visualizations:
- PERMA dimension predictions vs. ground truth
- Multi-topology graph attention weights
- Cross-cultural performance comparison
- Ablation study bar charts
- Hyperparameter sensitivity curves

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{mo2025perma,
  title={PERMA-Guided Multi-Topology Graph Neural Networks for Cross-Cultural Student Well-being Prediction},
  author={Mo, Lingqi and Zhang, Jie and Jiang, Zixiao and Wang, Shuanglei and Lee, ShiouYih},
  journal={PLOS ONE},
  year={2025},
  publisher={Public Library of Science}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Corresponding Author:**  
Jie Zhang  
Email: i24026180@student.newinti.edu.my

**First Author:**  
Lingqi Mo  
Email: molingqi123@163.com

For questions, issues, or collaborations, please:
- Open an issue in this repository
- Contact the corresponding author via email
- Visit our project website: [link_to_project_page]

## Acknowledgments

We thank:
- PLOS ONE reviewers for valuable feedback
- Kaggle for providing the Lifestyle and Wellbeing dataset
- The international university in Japan for the International Student Mental Health dataset
- INTI International University for computational resources

---

**Note**: This implementation is for research purposes. For production deployment in educational institutions, please contact the authors for guidance on ethical considerations and data privacy compliance.
