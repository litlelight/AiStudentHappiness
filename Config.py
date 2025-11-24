"""
Configuration File for PERMA-GNN-Transformer

This module contains all hyperparameters and configuration settings
as specified in the paper.

Paper References:
- Section 4.1.3: Model Parameter Settings
- Section 4.5: Hyperparameter Experiments
- Appendix A: Complete Hyperparameter Configurations (Table A1, A2)
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DatasetConfig:
    """
    Dataset configuration
    Paper Reference: Section 4.1.1 Dataset Description
    """
    # Lifestyle and Wellbeing Data (Large-scale, Western culture)
    lifestyle_path: str = "Dataset01.csv"
    lifestyle_n_samples: int = 12757
    lifestyle_n_features: int = 23
    lifestyle_culture: str = "Western"

    # International Student Mental Health Dataset (Small-scale, East Asian)
    mental_health_path: str = "Dataset02.zip"
    mental_health_n_samples: int = 268
    mental_health_culture: str = "East_Asian"

    # Data splitting (Section 4.1.2)
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # Random seed for reproducibility
    random_seed: int = 42

    # PERMA dimensions
    perma_dimensions: List[str] = field(default_factory=lambda: [
        'Positive_Emotion',
        'Engagement',
        'Relationships',
        'Meaning',
        'Achievement'
    ])


@dataclass
class ModelConfig:
    """
    Model architecture configuration
    Paper Reference: Section 4.1.3 Model Parameter Settings
    """
    # Input/Output dimensions
    input_dim: int = 23  # Lifestyle dataset features
    hidden_dim: int = 256  # Paper optimal: 256 (Section 4.5.3)
    embedding_dim: int = 128  # PERMA feature embedding dimension

    # Transformer configuration
    num_transformer_layers: int = 6  # Paper setting
    num_attention_heads: int = 5  # Paper optimal: 5 (corresponds to PERMA dimensions)
    attention_head_dim: int = 64  # hidden_dim / num_attention_heads (when hidden_dim=320)
    feedforward_dim: int = 1024  # Paper setting

    # Graph Neural Network configuration
    num_gcn_layers: int = 3  # Paper setting
    gcn_hidden_dim: int = 128  # Paper setting
    num_gat_heads: int = 8  # Paper setting
    gat_head_dim: int = 32  # Paper setting (gcn_hidden_dim / num_gat_heads when =256)

    # Multi-topology graph configuration
    num_graph_topologies: int = 4  # Cosine, Euclidean, Learning Style, PERMA-weighted

    # Dropout rates (Section 4.1.3)
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    feedforward_dropout: float = 0.1

    # Activation functions
    activation: str = 'gelu'  # Paper setting for Transformer
    gcn_activation: str = 'relu'  # For GNN layers

    # Output dimensions
    num_perma_dimensions: int = 5
    num_wellbeing_outputs: int = 1


@dataclass
class GraphConfig:
    """
    Graph construction configuration
    Paper Reference: Section 3.2.1 and Section 4.1.3
    """
    # Cosine Similarity Graph (Paper: threshold = 0.3)
    cosine_threshold: float = 0.3

    # Euclidean Distance k-NN Graph (Paper: k = 10)
    euclidean_k: int = 10

    # Learning Style Graph
    num_learning_styles: int = 4  # Visual, Auditory, Reading/Writing, Kinesthetic

    # PERMA-Weighted Graph (Paper: equal weights baseline)
    perma_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
    perma_threshold: float = 0.3

    # Edge normalization (Paper Section 4.1.3: symmetric normalization)
    edge_normalization: str = 'symmetric'  # D^(-1/2) A D^(-1/2)
    use_edge_weights: bool = True


@dataclass
class TrainingConfig:
    """
    Training configuration
    Paper Reference: Section 4.1.3 Model Parameter Settings
    """
    # Optimizer settings (Paper: AdamW)
    optimizer: str = 'adamw'
    learning_rate: float = 2e-4  # Paper optimal: 1e-4 or 2e-4 (Section 4.5.1)
    weight_decay: float = 1e-5  # Paper setting

    # AdamW parameters (Paper Appendix A, Table A1)
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # Learning rate scheduler (Paper: Cosine Annealing)
    scheduler_type: str = 'cosine'
    warmup_steps: int = 500  # Paper Appendix A
    min_learning_rate: float = 1e-6  # Paper setting

    # Training parameters
    num_epochs: int = 100  # Paper setting
    batch_size: int = 32  # Paper optimal: 32 (Section 4.5.2)

    # Gradient clipping (Paper Appendix A)
    gradient_clip_threshold: float = 1.0

    # Early stopping (Paper Section 4.1.2)
    early_stopping_patience: int = 15

    # Multi-task loss weights (Paper Section 3.2.4)
    lambda1: float = 1.0  # Overall wellbeing loss weight
    lambda2: float = 0.8  # PERMA dimension loss weight
    lambda3: float = 0.5  # Consistency loss weight


@dataclass
class EvaluationConfig:
    """
    Evaluation metrics configuration
    Paper Reference: Section 4.2 Evaluation Metrics
    """
    # PCE metric weights (Paper Section 4.2.2)
    pce_alpha: float = 0.3  # Weight for MAE
    pce_beta: float = 0.3  # Weight for RMSE
    pce_gamma: float = 0.4  # Weight for PERMA metrics (PDA + PCI)

    # Statistical significance testing
    significance_level: float = 0.01  # Paper: p < 0.01 for SOTA comparison
    confidence_interval: float = 0.95


@dataclass
class ExperimentConfig:
    """
    Experiment configuration
    Paper Reference: Section 4.1.2 Experimental Environment Configuration
    """
    # Hardware configuration (Paper Section 4.1.2)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_id: int = 0
    num_workers: int = 4
    pin_memory: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False

    # Experiment tracking
    experiment_name: str = "PERMA-GNN-Transformer"
    log_interval: int = 10  # Log every N batches
    save_checkpoint_interval: int = 10  # Save every N epochs

    # Results directory
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"


@dataclass
class HyperparameterSearchConfig:
    """
    Hyperparameter search space
    Paper Reference: Section 4.5 Hyperparameter Experiments
    """
    # Learning rate search space (Section 4.5.1)
    learning_rate_options: List[float] = field(default_factory=lambda: [
        1e-5, 5e-5, 1e-4, 5e-4, 1e-3
    ])

    # Batch size search space (Section 4.5.2)
    batch_size_options: List[int] = field(default_factory=lambda: [
        8, 16, 32, 64, 128
    ])

    # Hidden dimension search space (Section 4.5.3)
    hidden_dim_options: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 1024
    ])

    # Attention heads search space (Section 4.5.4)
    num_heads_options: List[int] = field(default_factory=lambda: [
        1, 3, 5, 8, 12
    ])


class Config:
    """
    Master configuration class that combines all config components
    """

    def __init__(
            self,
            dataset: str = "lifestyle",  # "lifestyle" or "mental_health"
            use_optimal: bool = True  # Use optimal hyperparameters from paper
    ):
        """
        Initialize configuration

        Args:
            dataset: Which dataset to use ("lifestyle" or "mental_health")
            use_optimal: Whether to use optimal hyperparameters from paper Section 4.5
        """
        self.dataset_type = dataset

        # Initialize all configuration components
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.graph = GraphConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.experiment = ExperimentConfig()
        self.hyperparameter_search = HyperparameterSearchConfig()

        # Adjust configuration based on dataset
        if dataset == "mental_health":
            self._configure_for_mental_health()

        # Apply optimal hyperparameters if requested
        if use_optimal:
            self._apply_optimal_hyperparameters()

    def _configure_for_mental_health(self):
        """
        Adjust configuration for Mental Health dataset (smaller scale)
        Paper: n=268 samples
        """
        # Smaller batch size for small dataset
        self.training.batch_size = 16

        # May need different number of input features
        # Adjust based on actual dataset structure
        # self.model.input_dim = XX  # Set based on actual features

        # Potentially adjust learning rate for smaller dataset
        # Keep other parameters the same for cross-cultural validation

    def _apply_optimal_hyperparameters(self):
        """
        Apply optimal hyperparameters found in paper Section 4.5

        Optimal configuration (from Section 4.5.5):
        - Learning rate: 1×10⁻⁴
        - Batch size: 32
        - Hidden dimension: 256
        - Attention heads: 5

        This configuration achieves:
        - MAE: 0.163 (32.9% improvement over baseline)
        - PCE: 0.792 (27.3% improvement over baseline)
        """
        self.training.learning_rate = 1e-4
        self.training.batch_size = 32
        self.model.hidden_dim = 256
        self.model.num_attention_heads = 5

        # Recalculate dependent dimensions
        self.model.attention_head_dim = self.model.hidden_dim // self.model.num_attention_heads

    def get_weight_init_config(self) -> Dict:
        """
        Get weight initialization configuration
        Paper Reference: Appendix A - Weight Initialization Strategies
        """
        return {
            'transformer_init': 'xavier_uniform',  # For Transformer layers
            'perma_embedding_init': 'xavier_uniform_with_priors',  # Theory-driven
            'gnn_init': 'kaiming_normal',  # For GNN layers
            'bias_init': 'zeros'
        }

    def get_normalization_config(self) -> Dict:
        """
        Get normalization configuration
        Paper Reference: Section 4.1.3
        """
        return {
            'graph_normalization': 'symmetric',  # D^(-1/2) A D^(-1/2)
            'layer_norm': True,
            'batch_norm': False
        }

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'dataset': self.dataset.__dict__,
            'model': self.model.__dict__,
            'graph': self.graph.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        config = cls()
        # Update configuration from loaded dict
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        return config

    def print_config(self):
        """Print configuration summary"""
        print("=" * 80)
        print("PERMA-GNN-Transformer Configuration")
        print("=" * 80)

        print("\n[Dataset Configuration]")
        print(f"  Dataset: {self.dataset_type}")
        print(f"  Features: {self.model.input_dim}")
        print(f"  Train/Val/Test: {self.dataset.train_ratio}/{self.dataset.val_ratio}/{self.dataset.test_ratio}")

        print("\n[Model Architecture]")
        print(f"  Hidden Dim: {self.model.hidden_dim}")
        print(f"  Embedding Dim: {self.model.embedding_dim}")
        print(f"  Transformer Layers: {self.model.num_transformer_layers}")
        print(f"  Attention Heads: {self.model.num_attention_heads} (PERMA-aligned)")
        print(f"  GCN Layers: {self.model.num_gcn_layers}")
        print(f"  GAT Heads: {self.model.num_gat_heads}")

        print("\n[Graph Construction]")
        print(f"  Cosine Threshold: {self.graph.cosine_threshold}")
        print(f"  Euclidean k: {self.graph.euclidean_k}")
        print(f"  Learning Styles: {self.graph.num_learning_styles}")
        print(f"  PERMA Weights: {self.graph.perma_weights}")

        print("\n[Training Configuration]")
        print(f"  Optimizer: {self.training.optimizer}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Weight Decay: {self.training.weight_decay}")
        print(f"  Gradient Clip: {self.training.gradient_clip_threshold}")

        print("\n[Multi-Task Loss Weights]")
        print(f"  λ1 (Wellbeing): {self.training.lambda1}")
        print(f"  λ2 (PERMA Dims): {self.training.lambda2}")
        print(f"  λ3 (Consistency): {self.training.lambda3}")

        print("\n[Evaluation Metrics]")
        print(
            f"  PCE Weights: α={self.evaluation.pce_alpha}, β={self.evaluation.pce_beta}, γ={self.evaluation.pce_gamma}")

        print("\n[Hardware]")
        print(f"  Device: {self.experiment.device}")
        print(f"  Random Seed: {self.experiment.seed}")

        print("=" * 80)


# Paper-reported optimal results for reference
PAPER_RESULTS = {
    'lifestyle': {  # Large-scale dataset (n=12,757)
        'MAE': 0.163,
        'RMSE': 0.215,
        'PDA': 0.841,
        'PCI': 0.798,
        'PCE': 0.792,
        'improvement_vs_best_baseline': 0.189  # 18.9%
    },
    'mental_health': {  # Small-scale dataset (n=268)
        'MAE': 0.148,
        'RMSE': 0.198,
        'PDA': 0.823,
        'PCI': 0.785,
        'PCE': 0.798,
        'improvement_vs_best_baseline': 0.278  # 27.8%
    }
}

# Baseline results for comparison (from paper Table 1)
BASELINE_RESULTS = {
    'Linear Regression': {
        'MAE': 0.356,
        'RMSE': 0.425,
        'PDA': 0.0,
        'PCI': 0.0,
        'PCE': 0.0
    },
    'Random Forest': {
        'MAE': 0.298,
        'RMSE': 0.372,
        'PDA': 0.0,
        'PCI': 0.0,
        'PCE': 0.0
    },
    'Gradient Boosting': {
        'MAE': 0.276,
        'RMSE': 0.351,
        'PDA': 0.0,
        'PCI': 0.0,
        'PCE': 0.0
    },
    'LSTM': {
        'MAE': 0.243,
        'RMSE': 0.312,
        'PDA': 0.648,
        'PCI': 0.621,
        'PCE': 0.634
    },
    'Transformer': {
        'MAE': 0.221,
        'RMSE': 0.285,
        'PDA': 0.661,
        'PCI': 0.637,
        'PCE': 0.649
    },
    'GraphSAGE': {
        'MAE': 0.218,
        'RMSE': 0.281,
        'PDA': 0.672,
        'PCI': 0.641,
        'PCE': 0.655
    },
    'King et al. (2024)': {  # State-of-the-art baseline
        'MAE': 0.201,
        'RMSE': 0.267,
        'PDA': 0.702,
        'PCI': 0.660,
        'PCE': 0.681
    },
    'Shahzad et al. (2024)': {  # State-of-the-art baseline
        'MAE': 0.208,
        'RMSE': 0.273,
        'PDA': 0.702,
        'PCI': 0.652,
        'PCE': 0.677
    }
}

if __name__ == "__main__":
    """
    Example usage of configuration system
    """

    print("PERMA-GNN-Transformer Configuration System")
    print("=" * 80)

    # Create configuration with optimal hyperparameters
    print("\n1. Creating optimal configuration for Lifestyle dataset...")
    config = Config(dataset="lifestyle", use_optimal=True)
    config.print_config()

    # Show paper results
    print("\n2. Paper-Reported Results:")
    print("-" * 80)
    print(f"Lifestyle Dataset (n={config.dataset.lifestyle_n_samples}):")
    for metric, value in PAPER_RESULTS['lifestyle'].items():
        if metric != 'improvement_vs_best_baseline':
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  Improvement vs Best Baseline: {value * 100:.1f}%")

    print(f"\nMental Health Dataset (n={config.dataset.mental_health_n_samples}):")
    for metric, value in PAPER_RESULTS['mental_health'].items():
        if metric != 'improvement_vs_best_baseline':
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  Improvement vs Best Baseline: {value * 100:.1f}%")

    # Show baseline comparison
    print("\n3. Baseline Methods Comparison (Large Dataset):")
    print("-" * 80)
    print(f"{'Method':<25} {'MAE':<8} {'RMSE':<8} {'PDA':<8} {'PCI':<8} {'PCE':<8}")
    print("-" * 80)

    for method, results in BASELINE_RESULTS.items():
        print(f"{method:<25} {results['MAE']:<8.3f} {results['RMSE']:<8.3f} "
              f"{results['PDA']:<8.3f} {results['PCI']:<8.3f} {results['PCE']:<8.3f}")

    # Our model
    our_results = PAPER_RESULTS['lifestyle']
    print(f"{'PERMA-GNN-Trans (Ours)':<25} {our_results['MAE']:<8.3f} "
          f"{our_results['RMSE']:<8.3f} {our_results['PDA']:<8.3f} "
          f"{our_results['PCI']:<8.3f} {our_results['PCE']:<8.3f}")

    # Save configuration
    print("\n4. Saving configuration to file...")
    config.save("config_lifestyle_optimal.json")
    print("✓ Saved to: config_lifestyle_optimal.json")

    # Create configuration for Mental Health dataset
    print("\n5. Creating configuration for Mental Health dataset...")
    config_mh = Config(dataset="mental_health", use_optimal=True)
    config_mh.save("config_mental_health_optimal.json")
    print("✓ Saved to: config_mental_health_optimal.json")

    # Show hyperparameter search space
    print("\n6. Hyperparameter Search Space (Paper Section 4.5):")
    print("-" * 80)
    print(f"Learning Rate: {config.hyperparameter_search.learning_rate_options}")
    print(f"Batch Size: {config.hyperparameter_search.batch_size_options}")
    print(f"Hidden Dim: {config.hyperparameter_search.hidden_dim_options}")
    print(f"Attention Heads: {config.hyperparameter_search.num_heads_options}")

    print("\n" + "=" * 80)
    print("Configuration System Demo Complete!")
    print("=" * 80)