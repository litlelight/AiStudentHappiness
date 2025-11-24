"""
Evaluation Metrics Module for PERMA-GNN-Transformer

This module implements the comprehensive evaluation framework described in the paper:
- Traditional regression metrics: MAE, RMSE
- PERMA theory-driven metrics: PDA, PCI, PCE

Paper Reference: Section 4.2 Evaluation Metrics
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from scipy import stats


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for student well-being prediction

    Implements:
    1. Mean Absolute Error (MAE) - Section 4.2.1
    2. Root Mean Square Error (RMSE) - Section 4.2.1
    3. PERMA Dimension Accuracy (PDA) - Section 4.2.2
    4. PERMA Consistency Index (PCI) - Section 4.2.2
    5. PERMA Comprehensive Evaluation (PCE) - Section 4.2.2
    """

    def __init__(self):
        """Initialize evaluation metrics calculator"""
        pass

    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE)

        Formula (from paper Section 4.2.1):
            MAE = (1/n) Σ|y_i - ŷ_i|

        Args:
            y_true: True wellbeing labels [n_samples, 1] or [n_samples]
            y_pred: Predicted wellbeing values [n_samples, 1] or [n_samples]

        Returns:
            MAE value (lower is better)
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        mae = np.mean(np.abs(y_true - y_pred))
        return float(mae)

    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Square Error (RMSE)

        Formula (from paper Section 4.2.1):
            RMSE = sqrt((1/n) Σ(y_i - ŷ_i)²)

        RMSE assigns higher weights to larger errors, making it sensitive
        to outliers and prediction failures in extreme cases.

        Args:
            y_true: True wellbeing labels [n_samples, 1] or [n_samples]
            y_pred: Predicted wellbeing values [n_samples, 1] or [n_samples]

        Returns:
            RMSE value (lower is better)
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)

    @staticmethod
    def compute_pda(perma_true: np.ndarray, perma_pred: np.ndarray) -> float:
        """
        PERMA Dimension Accuracy (PDA)

        Formula (from paper Section 4.2.2):
            PDA = 1 - (1/(n×5)) ΣΣ|y_i^(p) - ŷ_i^(p)|

        where y_i^(p) is the true label for the p-th PERMA dimension
        of student i, and ŷ_i^(p) is the predicted value.

        PDA evaluates the model's individual prediction accuracy for the
        five dimensions: Positive emotions, Engagement, Relationships,
        Meaning, and Achievement.

        Args:
            perma_true: True PERMA dimension labels [n_samples, 5]
            perma_pred: Predicted PERMA dimension values [n_samples, 5]

        Returns:
            PDA score in [0, 1] (higher is better)
        """
        # Compute MAE for each dimension
        dimension_mae = np.mean(np.abs(perma_true - perma_pred))

        # Convert to accuracy metric (1 - normalized error)
        pda = 1.0 - dimension_mae

        # Ensure PDA is in [0, 1]
        pda = np.clip(pda, 0.0, 1.0)

        return float(pda)

    @staticmethod
    def compute_pci(wellbeing_true: np.ndarray,
                    wellbeing_pred: np.ndarray,
                    perma_pred: np.ndarray) -> float:
        """
        PERMA Consistency Index (PCI)

        Formula (from paper Section 4.2.2):
            PCI = 1 - (1/n) Σ|ŷ_i - (1/5)Σŷ_i^(p)|

        PCI evaluates the theoretical consistency between overall wellbeing
        prediction and PERMA dimension predictions. According to PERMA theory,
        overall wellbeing should be consistent with the average level of
        each dimension.

        This metric ensures that model prediction results conform to the
        basic assumption in PERMA theory.

        Args:
            wellbeing_true: True wellbeing labels [n_samples, 1] or [n_samples]
            wellbeing_pred: Predicted wellbeing [n_samples, 1] or [n_samples]
            perma_pred: Predicted PERMA dimensions [n_samples, 5]

        Returns:
            PCI score in [0, 1] (higher is better)
        """
        wellbeing_pred = wellbeing_pred.flatten()

        # Compute mean of PERMA dimension predictions
        perma_mean = np.mean(perma_pred, axis=1)

        # Compute consistency error
        consistency_error = np.mean(np.abs(wellbeing_pred - perma_mean))

        # Convert to consistency metric (1 - error)
        pci = 1.0 - consistency_error

        # Ensure PCI is in [0, 1]
        pci = np.clip(pci, 0.0, 1.0)

        return float(pci)

    @staticmethod
    def compute_pce(mae: float,
                    rmse: float,
                    pda: float,
                    pci: float,
                    alpha: float = 0.3,
                    beta: float = 0.3,
                    gamma: float = 0.4) -> float:
        """
        PERMA Comprehensive Evaluation (PCE)

        Formula (from paper Section 4.2.2):
            PCE = α·(1-MAE_norm) + β·(1-RMSE_norm) + γ·(PDA + PCI)/2

        where α, β, γ are weight parameters, and MAE_norm, RMSE_norm are
        normalized MAE and RMSE values.

        PCE provides a comprehensive evaluation reflecting model performance
        at multiple levels: accuracy, consistency, and theoretical compliance.

        Weight parameters (from paper):
        - α = 0.3: weight for MAE
        - β = 0.3: weight for RMSE
        - γ = 0.4: weight for PERMA theory metrics

        Args:
            mae: Mean Absolute Error
            rmse: Root Mean Square Error
            pda: PERMA Dimension Accuracy
            pci: PERMA Consistency Index
            alpha: Weight for MAE (default 0.3)
            beta: Weight for RMSE (default 0.3)
            gamma: Weight for PERMA metrics (default 0.4)

        Returns:
            PCE score in [0, 1] (higher is better)
        """
        # Normalize MAE and RMSE (assume max reasonable error is 1.0 for normalized data)
        mae_norm = np.clip(mae, 0.0, 1.0)
        rmse_norm = np.clip(rmse, 0.0, 1.0)

        # Compute PCE
        pce = (alpha * (1.0 - mae_norm) +
               beta * (1.0 - rmse_norm) +
               gamma * (pda + pci) / 2.0)

        # Ensure PCE is in [0, 1]
        pce = np.clip(pce, 0.0, 1.0)

        return float(pce)

    @staticmethod
    def compute_all_metrics(wellbeing_true: np.ndarray,
                            wellbeing_pred: np.ndarray,
                            perma_true: np.ndarray,
                            perma_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all evaluation metrics at once

        Args:
            wellbeing_true: True wellbeing labels [n_samples, 1] or [n_samples]
            wellbeing_pred: Predicted wellbeing [n_samples, 1] or [n_samples]
            perma_true: True PERMA dimensions [n_samples, 5]
            perma_pred: Predicted PERMA dimensions [n_samples, 5]

        Returns:
            Dictionary containing all metrics
        """
        # Traditional metrics
        mae = EvaluationMetrics.compute_mae(wellbeing_true, wellbeing_pred)
        rmse = EvaluationMetrics.compute_rmse(wellbeing_true, wellbeing_pred)

        # PERMA theory metrics
        pda = EvaluationMetrics.compute_pda(perma_true, perma_pred)
        pci = EvaluationMetrics.compute_pci(wellbeing_true, wellbeing_pred, perma_pred)

        # Comprehensive metric
        pce = EvaluationMetrics.compute_pce(mae, rmse, pda, pci)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'PDA': pda,
            'PCI': pci,
            'PCE': pce
        }

    @staticmethod
    def compute_per_dimension_metrics(perma_true: np.ndarray,
                                      perma_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each individual PERMA dimension

        This provides fine-grained analysis of model performance on:
        - P: Positive emotions
        - E: Engagement
        - R: Relationships
        - M: Meaning
        - A: Achievement

        Args:
            perma_true: True PERMA dimensions [n_samples, 5]
            perma_pred: Predicted PERMA dimensions [n_samples, 5]

        Returns:
            Dictionary with metrics for each dimension
        """
        dimension_names = ['Positive_Emotion', 'Engagement', 'Relationships',
                           'Meaning', 'Achievement']

        per_dim_metrics = {}

        for i, dim_name in enumerate(dimension_names):
            dim_true = perma_true[:, i]
            dim_pred = perma_pred[:, i]

            per_dim_metrics[dim_name] = {
                'MAE': EvaluationMetrics.compute_mae(dim_true, dim_pred),
                'RMSE': EvaluationMetrics.compute_rmse(dim_true, dim_pred),
                'Correlation': np.corrcoef(dim_true, dim_pred)[0, 1]
            }

        return per_dim_metrics


class StatisticalSignificanceTest:
    """
    Statistical significance testing for model comparisons

    Implements paired t-test as described in paper Section 4.2.3
    """

    @staticmethod
    def paired_t_test(errors_model_a: np.ndarray,
                      errors_model_b: np.ndarray) -> Tuple[float, float]:
        """
        Paired t-test for comparing two models

        Formula (from paper Section 4.2.3):
            t = (d̄ - 0) / (s_d / sqrt(n))

        where d̄ is the mean of error differences, s_d is the standard
        deviation, and n is the sample size.

        Args:
            errors_model_a: Prediction errors from model A [n_samples]
            errors_model_b: Prediction errors from model B [n_samples]

        Returns:
            t_statistic: t-test statistic
            p_value: p-value for statistical significance
                     p < 0.001: extremely significant
                     p < 0.01: highly significant
                     p < 0.05: significant
        """
        # Compute error differences
        error_diff = errors_model_a - errors_model_b

        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(errors_model_a, errors_model_b)

        return float(t_statistic), float(p_value)

    @staticmethod
    def compute_confidence_interval(metric_values: np.ndarray,
                                    confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute confidence interval for a metric

        Args:
            metric_values: Array of metric values from multiple runs
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            mean: Mean value
            lower_bound: Lower bound of CI
            upper_bound: Upper bound of CI
        """
        mean = np.mean(metric_values)
        std_err = stats.sem(metric_values)

        # Compute confidence interval
        ci = std_err * stats.t.ppf((1 + confidence) / 2, len(metric_values) - 1)

        return float(mean), float(mean - ci), float(mean + ci)

    @staticmethod
    def interpret_p_value(p_value: float) -> str:
        """
        Interpret p-value significance level

        From paper Section 4.2.3:
        - p < 0.001: extremely high statistical significance
        - p < 0.01: high statistical significance
        - p < 0.05: statistical significance
        - p >= 0.05: insufficient evidence

        Args:
            p_value: p-value from statistical test

        Returns:
            Interpretation string
        """
        if p_value < 0.001:
            return "Extremely Significant (p < 0.001) ***"
        elif p_value < 0.01:
            return "Highly Significant (p < 0.01) **"
        elif p_value < 0.05:
            return "Significant (p < 0.05) *"
        else:
            return "Not Significant (p >= 0.05)"


class ResultsComparison:
    """
    Comparison of model results with baselines

    Implements the comparative analysis framework from paper Section 4.3
    """

    @staticmethod
    def compare_with_baselines(
            model_metrics: Dict[str, float],
            baseline_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model performance with baseline methods

        Args:
            model_metrics: Metrics for the proposed model
            baseline_metrics: Dictionary of {method_name: metrics}

        Returns:
            Dictionary with improvement percentages for each baseline
        """
        comparisons = {}

        for baseline_name, baseline_vals in baseline_metrics.items():
            improvements = {}

            for metric_name in ['MAE', 'RMSE']:
                # For error metrics, lower is better
                if metric_name in model_metrics and metric_name in baseline_vals:
                    baseline_val = baseline_vals[metric_name]
                    model_val = model_metrics[metric_name]

                    # Improvement percentage
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                    improvements[f'{metric_name}_improvement'] = improvement

            for metric_name in ['PDA', 'PCI', 'PCE']:
                # For accuracy metrics, higher is better
                if metric_name in model_metrics and metric_name in baseline_vals:
                    baseline_val = baseline_vals[metric_name]
                    model_val = model_metrics[metric_name]

                    # Improvement percentage
                    improvement = ((model_val - baseline_val) / baseline_val) * 100
                    improvements[f'{metric_name}_improvement'] = improvement

            comparisons[baseline_name] = improvements

        return comparisons

    @staticmethod
    def format_results_table(model_name: str,
                             metrics: Dict[str, float],
                             include_header: bool = True) -> str:
        """
        Format results as a table row for paper

        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            include_header: Whether to include table header

        Returns:
            Formatted table string
        """
        table = ""

        if include_header:
            table += "| Model | MAE | RMSE | PDA | PCI | PCE |\n"
            table += "|-------|-----|------|-----|-----|-----|\n"

        table += f"| {model_name:<20} | "
        table += f"{metrics.get('MAE', 0):.3f} | "
        table += f"{metrics.get('RMSE', 0):.3f} | "
        table += f"{metrics.get('PDA', 0):.3f} | "
        table += f"{metrics.get('PCI', 0):.3f} | "
        table += f"{metrics.get('PCE', 0):.3f} |\n"

        return table


class PerformanceTracker:
    """Track model performance during training"""

    def __init__(self):
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = None
        self.best_val_metrics = None
        self.best_epoch = -1

    def update(self, epoch: int,
               train_metrics: Dict[str, float],
               val_metrics: Dict[str, float]):
        """
        Update metrics for an epoch

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)

        # Track best validation performance
        if self.best_val_metrics is None or val_metrics['PCE'] > self.best_val_metrics['PCE']:
            self.best_val_metrics = val_metrics.copy()
            self.best_epoch = epoch

    def set_test_metrics(self, test_metrics: Dict[str, float]):
        """Set final test metrics"""
        self.test_metrics = test_metrics

    def get_summary(self) -> Dict:
        """Get summary of training"""
        return {
            'best_epoch': self.best_epoch,
            'best_val_metrics': self.best_val_metrics,
            'final_test_metrics': self.test_metrics,
            'total_epochs': len(self.train_metrics)
        }

    def print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)

        if self.best_val_metrics:
            print(f"\nBest Validation Performance (Epoch {self.best_epoch}):")
            for metric, value in self.best_val_metrics.items():
                print(f"  {metric}: {value:.4f}")

        if self.test_metrics:
            print(f"\nFinal Test Performance:")
            for metric, value in self.test_metrics.items():
                print(f"  {metric}: {value:.4f}")

        print("\n" + "=" * 80)


def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate model on a dataset

    Args:
        model: PERMA-GNN-Transformer model
        data_loader: PyTorch DataLoader
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_wellbeing_true = []
    all_wellbeing_pred = []
    all_perma_true = []
    all_perma_pred = []

    with torch.no_grad():
        for batch in data_loader:
            # Unpack batch
            features = batch['features'].to(device)
            edge_index_list = [ei.to(device) for ei in batch['edge_index_list']]
            wellbeing_true = batch['wellbeing'].to(device)
            perma_true = batch['perma'].to(device)

            # Forward pass
            wellbeing_pred, perma_pred = model(features, edge_index_list)

            # Collect predictions
            all_wellbeing_true.append(wellbeing_true.cpu().numpy())
            all_wellbeing_pred.append(wellbeing_pred.cpu().numpy())
            all_perma_true.append(perma_true.cpu().numpy())
            all_perma_pred.append(perma_pred.cpu().numpy())

    # Concatenate all batches
    wellbeing_true = np.concatenate(all_wellbeing_true, axis=0)
    wellbeing_pred = np.concatenate(all_wellbeing_pred, axis=0)
    perma_true = np.concatenate(all_perma_true, axis=0)
    perma_pred = np.concatenate(all_perma_pred, axis=0)

    # Compute metrics
    metrics = EvaluationMetrics.compute_all_metrics(
        wellbeing_true, wellbeing_pred,
        perma_true, perma_pred
    )

    return metrics


if __name__ == "__main__":
    """
    Example usage demonstrating evaluation metrics computation
    """

    print("PERMA-GNN-Transformer: Evaluation Metrics Demonstration")
    print("=" * 80)

    # Simulate prediction results
    np.random.seed(42)
    n_samples = 100

    # Generate synthetic true labels
    wellbeing_true = np.random.uniform(0, 1, (n_samples, 1))
    perma_true = np.random.uniform(0, 1, (n_samples, 5))

    # Simulate predictions with some noise
    wellbeing_pred = wellbeing_true + np.random.normal(0, 0.1, (n_samples, 1))
    perma_pred = perma_true + np.random.normal(0, 0.1, (n_samples, 5))

    # Clip to [0, 1]
    wellbeing_pred = np.clip(wellbeing_pred, 0, 1)
    perma_pred = np.clip(perma_pred, 0, 1)

    print("\n--- Computing All Metrics ---")
    metrics = EvaluationMetrics.compute_all_metrics(
        wellbeing_true, wellbeing_pred,
        perma_true, perma_pred
    )

    print("\nOverall Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Per-dimension analysis
    print("\n--- Per-Dimension Analysis ---")
    per_dim_metrics = EvaluationMetrics.compute_per_dimension_metrics(
        perma_true, perma_pred
    )

    for dim_name, dim_metrics in per_dim_metrics.items():
        print(f"\n{dim_name}:")
        for metric_name, value in dim_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Statistical significance test
    print("\n--- Statistical Significance Test ---")

    # Simulate errors from two models
    errors_proposed = np.abs(wellbeing_true - wellbeing_pred).flatten()
    errors_baseline = np.abs(
        wellbeing_true - (wellbeing_true + np.random.normal(0, 0.15, wellbeing_true.shape))).flatten()

    t_stat, p_value = StatisticalSignificanceTest.paired_t_test(
        errors_baseline, errors_proposed
    )

    interpretation = StatisticalSignificanceTest.interpret_p_value(p_value)

    print(f"\nComparing Proposed Model vs Baseline:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Interpretation: {interpretation}")

    # Results comparison
    print("\n--- Comparison with Baselines ---")

    # Example baseline metrics (from paper Table 1)
    baseline_metrics = {
        'Linear Regression': {'MAE': 0.356, 'RMSE': 0.425, 'PDA': 0.0, 'PCI': 0.0, 'PCE': 0.0},
        'Transformer': {'MAE': 0.221, 'RMSE': 0.285, 'PDA': 0.661, 'PCI': 0.637, 'PCE': 0.649},
        'King et al. (2024)': {'MAE': 0.201, 'RMSE': 0.267, 'PDA': 0.702, 'PCI': 0.660, 'PCE': 0.681}
    }

    # Our model (paper results: MAE=0.163, PCE=0.792)
    model_metrics = {
        'MAE': 0.163,
        'RMSE': 0.215,
        'PDA': 0.841,
        'PCI': 0.798,
        'PCE': 0.792
    }

    comparisons = ResultsComparison.compare_with_baselines(
        model_metrics, baseline_metrics
    )

    for baseline_name, improvements in comparisons.items():
        print(f"\nImprovement over {baseline_name}:")
        for metric_name, improvement in improvements.items():
            if improvement > 0:
                print(f"  {metric_name}: +{improvement:.1f}%")
            else:
                print(f"  {metric_name}: {improvement:.1f}%")

    # Format results table
    print("\n--- Results Table (Paper Format) ---\n")

    table = ResultsComparison.format_results_table(
        "Linear Regression", baseline_metrics['Linear Regression'], include_header=True
    )
    table += ResultsComparison.format_results_table(
        "Transformer", baseline_metrics['Transformer'], include_header=False
    )
    table += ResultsComparison.format_results_table(
        "King et al. (2024)", baseline_metrics['King et al. (2024)'], include_header=False
    )
    table += ResultsComparison.format_results_table(
        "PERMA-GNN-Trans (Ours)", model_metrics, include_header=False
    )

    print(table)

    print("=" * 80)
    print("Evaluation Metrics Demonstration Complete!")
    print("=" * 80)
