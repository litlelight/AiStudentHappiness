"""
PERMA-GNN-Transformer: Cross-Cultural Student Well-being Prediction Model

This file contains the core architecture of the PERMA-GNN-Transformer model.
For complete implementation details, please refer to the published paper.

Paper: "PERMA-Guided Multi-Topology Graph Neural Networks for Cross-Cultural
        Student Well-being Prediction" (PLOS ONE, 2025)

Author: Lingqi Mo, Jie Zhang, et al.
Contact: i24026180@student.newinti.edu.my
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from typing import List, Tuple, Optional
import math


# ============================================================================
# PERMA Theory-Driven Feature Embedding Module
# ============================================================================

class PERMAFeatureEmbedding(nn.Module):
    """
    Maps raw student features to PERMA psychological dimensions.

    Architecture:
        Input (d-dim) → PERMA Projection → 5 dimensions × embedding_dim

    Innovation: Theory-driven weight initialization based on psychological priors.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super(PERMAFeatureEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Projection layers for each PERMA dimension
        self.perma_projections = nn.ModuleDict({
            'positive_emotion': nn.Linear(input_dim, embedding_dim),
            'engagement': nn.Linear(input_dim, embedding_dim),
            'relationships': nn.Linear(input_dim, embedding_dim),
            'meaning': nn.Linear(input_dim, embedding_dim),
            'achievement': nn.Linear(input_dim, embedding_dim)
        })

        # Cross-modal attention for PERMA dimension interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=5,
            dropout=0.1
        )

        # Initialize with psychological priors
        self._initialize_perma_weights()

    def _initialize_perma_weights(self):
        """
        Initialize projection weights based on PERMA psychological theory.

        Implementation details: See Section 3.2.2 of the paper.
        This method assigns higher initial weights to features that are
        theoretically relevant to each PERMA dimension.

        NOTE: Specific weight initialization strategy is omitted.
              For full details, refer to the published paper.
        """
        # TODO: Custom initialization based on feature-PERMA mapping
        # Hint: Social features → Relationships, Academic features → Achievement
        for name, layer in self.perma_projections.items():
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            # Advanced initialization omitted - see paper Section 3.2.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Student features [batch_size, input_dim]

        Returns:
            PERMA embeddings [batch_size, 5, embedding_dim]
        """
        perma_dims = []
        for dim_name in ['positive_emotion', 'engagement', 'relationships',
                         'meaning', 'achievement']:
            perma_dims.append(self.perma_projections[dim_name](x))

        # Stack: [batch_size, 5, embedding_dim]
        perma_features = torch.stack(perma_dims, dim=1)

        # Apply cross-modal attention (implementation simplified)
        # Full implementation includes residual connections - see paper
        attended, _ = self.cross_attention(
            perma_features, perma_features, perma_features
        )

        return perma_features + attended  # Residual connection


# ============================================================================
# Multi-Topology Graph Construction (Interface Only)
# ============================================================================

class GraphConstructor:
    """
    Builds four types of student relationship graphs.

    Graph Types:
        1. Cosine Similarity Graph
        2. Euclidean Distance Graph
        3. Learning Style Graph
        4. PERMA-Weighted Graph (theory-driven)

    NOTE: Implementation details are omitted for competitive reasons.
          Core algorithms described in paper Section 3.2.1.
    """

    @staticmethod
    def construct_cosine_graph(features: torch.Tensor,
                               threshold: float = 0.3) -> torch.Tensor:
        """Cosine similarity-based adjacency matrix"""
        # Simplified implementation - production version has optimizations
        norm_features = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(norm_features, norm_features.t())
        adjacency = (similarity > threshold).float()
        return adjacency

    @staticmethod
    def construct_euclidean_graph(features: torch.Tensor,
                                  k: int = 10) -> torch.Tensor:
        """Euclidean distance k-NN graph"""
        # Implementation omitted - see paper for distance metric details
        raise NotImplementedError("See paper Section 3.2.1")

    @staticmethod
    def construct_learning_style_graph(learning_styles: torch.Tensor) -> torch.Tensor:
        """Discrete learning style-based graph"""
        # Requires learning style feature engineering - omitted
        raise NotImplementedError("Refer to paper for learning style encoding")

    @staticmethod
    def construct_perma_weighted_graph(perma_features: torch.Tensor,
                                       weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        PERMA theory-driven weighted graph.

        Formula (from paper):
            a_ij^(PERMA) = Σ(p=1 to 5) w_p · sim_p(x_i^(p), x_j^(p))

        Args:
            perma_features: [num_students, 5, dim] PERMA embeddings
            weights: Theory-based dimension weights (default: equal weights)

        NOTE: Optimal weight computation strategy is proprietary.
              See paper Section 3.2.1 for theoretical justification.
        """
        if weights is None:
            weights = [0.2] * 5  # Simplified equal weighting

        # Advanced weighting strategy omitted
        # Production version uses adaptive weights based on cultural context
        raise NotImplementedError("Full implementation in paper Section 3.2.1")


# ============================================================================
# Multi-Topology Graph Neural Network
# ============================================================================

class GraphLevelAttention(nn.Module):
    """
    Fuses information from multiple graph topologies using attention.

    Innovation: Learning style and stress-aware attention weights.
    Formula (paper Section 3.2.3):
        β_k = softmax(v_k^T tanh(W_g [h_style | h_stress] + b_g))
    """

    def __init__(self, hidden_dim: int, num_topologies: int = 4):
        super(GraphLevelAttention, self).__init__()
        self.num_topologies = num_topologies

        # Attention scoring network
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # [style | stress] concat
            nn.Tanh(),
            nn.Linear(hidden_dim, num_topologies),
            nn.Softmax(dim=-1)
        )

    def forward(self, graph_outputs: List[torch.Tensor],
                style_embedding: torch.Tensor,
                stress_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_outputs: List of 4 graph representations [batch, hidden_dim]
            style_embedding: Learning style features [batch, hidden_dim]
            stress_embedding: Stress level features [batch, hidden_dim]

        Returns:
            Fused graph representation [batch, hidden_dim]
        """
        # Concatenate style and stress for attention context
        context = torch.cat([style_embedding, stress_embedding], dim=-1)

        # Compute attention weights [batch, 4]
        attention_weights = self.attention_fc(context)

        # Weighted sum of graph outputs
        stacked_outputs = torch.stack(graph_outputs, dim=1)  # [batch, 4, hidden]
        fused = torch.einsum('bk,bkd->bd', attention_weights, stacked_outputs)

        return fused


class MultiTopologyGNN(nn.Module):
    """
    Processes student relationships through 4 parallel graph topologies.

    Architecture (per topology):
        Input → GCN (3 layers) → GAT (8 heads) → Graph-level Attention Fusion

    NOTE: Layer-specific configurations and normalization strategies omitted.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8):
        super(MultiTopologyGNN, self).__init__()
        self.num_topologies = 4

        # GCN layers for each topology
        self.gcn_layers = nn.ModuleList([
            nn.ModuleList([
                GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(3)
            ]) for _ in range(self.num_topologies)
        ])

        # GAT layers for each topology
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(self.num_topologies)
        ])

        # Graph-level attention fusion
        self.graph_fusion = GraphLevelAttention(hidden_dim, self.num_topologies)

        # Normalization layers (specific configuration omitted)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.num_topologies)
        ])

    def forward(self, x: torch.Tensor,
                edge_index_list: List[torch.Tensor],
                style_features: torch.Tensor,
                stress_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index_list: List of 4 edge indices for different topologies
            style_features: Learning style embeddings [num_nodes, hidden_dim]
            stress_features: Stress level embeddings [num_nodes, hidden_dim]

        Returns:
            Fused node representations [num_nodes, hidden_dim]
        """
        topology_outputs = []

        for k in range(self.num_topologies):
            h = x

            # GCN propagation (3 layers)
            for gcn_layer in self.gcn_layers[k]:
                h = gcn_layer(h, edge_index_list[k])
                h = F.relu(h)
                h = F.dropout(h, p=0.1, training=self.training)

            # GAT attention
            h = self.gat_layers[k](h, edge_index_list[k])
            h = self.layer_norms[k](h)

            topology_outputs.append(h)

        # Fuse across topologies with attention
        fused_output = self.graph_fusion(
            topology_outputs, style_features, stress_features
        )

        return fused_output


# ============================================================================
# PERMA-Aligned Transformer Encoder
# ============================================================================

class PERMATransformerEncoder(nn.Module):
    """
    5-head Transformer with attention heads aligned to PERMA dimensions.

    Innovation: Each attention head specializes in one PERMA dimension.
    - Head 1 → Positive emotions
    - Head 2 → Engagement
    - Head 3 → Relationships
    - Head 4 → Meaning
    - Head 5 → Achievement
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 6):
        super(PERMATransformerEncoder, self).__init__()
        assert hidden_dim % 5 == 0, "hidden_dim must be divisible by 5 (PERMA heads)"

        self.num_perma_heads = 5
        self.head_dim = hidden_dim // 5

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.num_perma_heads,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # PERMA-specific positional encodings (implementation omitted)
        # See paper Section 3.2.4 for encoding strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]

        Returns:
            Encoded features [batch_size, seq_len, hidden_dim]
        """
        # Add PERMA-aware positional encoding (omitted)
        # Production version includes dimension-specific position embeddings

        encoded = self.transformer(x)
        return encoded


# ============================================================================
# Complete PERMA-GNN-Transformer Model
# ============================================================================

class PERMAGNNTransformer(nn.Module):
    """
    Complete model integrating PERMA theory, GNN, and Transformer.

    Pipeline:
        Raw Features → PERMA Embedding → Multi-Topology GNN →
        PERMA-Transformer → Multi-Task Prediction (Wellbeing + 5 PERMA dims)

    Performance (paper results):
        - Large dataset (n=12,757): MAE=0.163, PCE=0.792
        - Small dataset (n=268): MAE=0.148, PCE=0.798
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 embedding_dim: int = 128,
                 num_transformer_layers: int = 6):
        super(PERMAGNNTransformer, self).__init__()

        # Stage 1: PERMA Feature Embedding
        self.perma_embedding = PERMAFeatureEmbedding(input_dim, embedding_dim)

        # Stage 2: Multi-Topology GNN
        self.multi_gnn = MultiTopologyGNN(
            input_dim=embedding_dim * 5,  # 5 PERMA dims concatenated
            hidden_dim=hidden_dim
        )

        # Stage 3: PERMA-Aligned Transformer
        self.transformer = PERMATransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers
        )

        # Stage 4: Multi-Task Prediction Heads
        self.wellbeing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.perma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 5),
            nn.Sigmoid()
        )

        # Additional feature extractors (simplified)
        self.style_extractor = nn.Linear(input_dim, hidden_dim)
        self.stress_extractor = nn.Linear(input_dim, hidden_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            x: Student features [batch_size, input_dim]
            edge_index_list: List of 4 edge indices for graph topologies

        Returns:
            wellbeing_pred: Overall wellbeing [batch_size, 1]
            perma_preds: 5 PERMA dimensions [batch_size, 5]
        """
        batch_size = x.size(0)

        # Stage 1: PERMA embedding
        perma_features = self.perma_embedding(x)  # [batch, 5, emb_dim]
        perma_flat = perma_features.view(batch_size, -1)  # [batch, 5*emb_dim]

        # Extract style and stress features (simplified)
        style_emb = F.relu(self.style_extractor(x))
        stress_emb = F.relu(self.stress_extractor(x))

        # Stage 2: Multi-topology GNN processing
        graph_output = self.multi_gnn(
            perma_flat, edge_index_list, style_emb, stress_emb
        )  # [batch, hidden_dim]

        # Stage 3: Transformer encoding
        # Reshape for transformer: [batch, seq_len=1, hidden_dim]
        transformer_input = graph_output.unsqueeze(1)
        encoded = self.transformer(transformer_input)  # [batch, 1, hidden_dim]
        encoded = encoded.squeeze(1)  # [batch, hidden_dim]

        # Stage 4: Multi-task predictions
        wellbeing_pred = self.wellbeing_head(encoded)  # [batch, 1]
        perma_preds = self.perma_head(encoded)  # [batch, 5]

        return wellbeing_pred, perma_preds

    def compute_loss(self,
                     wellbeing_pred: torch.Tensor,
                     perma_preds: torch.Tensor,
                     wellbeing_target: torch.Tensor,
                     perma_targets: torch.Tensor,
                     lambda1: float = 1.0,
                     lambda2: float = 0.8,
                     lambda3: float = 0.5) -> torch.Tensor:
        """
        Multi-task loss with PERMA consistency constraint.

        Loss = λ1·L_wellbeing + λ2·L_PERMA + λ3·L_consistency

        NOTE: Advanced loss balancing strategies (e.g., uncertainty weighting,
              dynamic lambda scheduling) are omitted.
              See paper Section 3.2.4 for full details.
        """
        # Wellbeing prediction loss
        loss_wellbeing = F.mse_loss(wellbeing_pred, wellbeing_target)

        # PERMA dimension prediction loss
        loss_perma = F.mse_loss(perma_preds, perma_targets)

        # PERMA consistency loss (theory constraint)
        # Constraint: overall wellbeing ≈ mean(PERMA dimensions)
        perma_mean = perma_preds.mean(dim=1, keepdim=True)
        loss_consistency = F.mse_loss(wellbeing_pred, perma_mean)

        # Total loss (simplified version)
        # Production version includes additional regularization terms
        total_loss = (lambda1 * loss_wellbeing +
                      lambda2 * loss_perma +
                      lambda3 * loss_consistency)

        return total_loss


# ============================================================================
# Model Factory (Simplified)
# ============================================================================

def create_perma_model(input_dim: int,
                       config: Optional[dict] = None) -> PERMAGNNTransformer:
    """
    Factory function to create PERMA-GNN-Transformer model.

    Args:
        input_dim: Number of input features (23 for Lifestyle, varies by dataset)
        config: Model configuration dict (uses defaults if None)

    Returns:
        Initialized PERMA-GNN-Transformer model

    Example:
        >>> model = create_perma_model(input_dim=23)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    """
    if config is None:
        config = {
            'hidden_dim': 256,
            'embedding_dim': 128,
            'num_transformer_layers': 6
        }

    model = PERMAGNNTransformer(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        num_transformer_layers=config['num_transformer_layers']
    )

    return model


# ============================================================================
# Usage Example (Pseudo-code)
# ============================================================================

if __name__ == "__main__":
    """
    Minimal usage example. 

    NOTE: Full training pipeline requires additional components:
        - Data preprocessing (feature engineering, normalization)
        - Graph construction (4 topology builders)
        - Training loop (early stopping, learning rate scheduling)
        - Evaluation metrics (MAE, RMSE, PDA, PCI, PCE)

    For complete implementation, please refer to the published paper or
    contact the authors for collaboration opportunities.
    """

    # Dummy data
    batch_size = 32
    num_nodes = 100
    input_dim = 23  # Lifestyle dataset

    # Create model
    model = create_perma_model(input_dim=input_dim)

    # Dummy inputs
    x = torch.randn(batch_size, input_dim)

    # Placeholder for edge indices (requires graph construction)
    # In practice, use GraphConstructor class with real student data
    edge_index_list = [
        torch.randint(0, batch_size, (2, 100)) for _ in range(4)
    ]

    # Forward pass
    wellbeing_pred, perma_preds = model(x, edge_index_list)

    print(f"Wellbeing predictions: {wellbeing_pred.shape}")  # [32, 1]
    print(f"PERMA predictions: {perma_preds.shape}")  # [32, 5]

    # Dummy targets
    wellbeing_target = torch.rand(batch_size, 1)
    perma_targets = torch.rand(batch_size, 5)

    # Compute loss
    loss = model.compute_loss(
        wellbeing_pred, perma_preds,
        wellbeing_target, perma_targets
    )

    print(f"Training loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("NOTE: This is a simplified demonstration.")
    print("For full training pipeline and optimal hyperparameters,")
    print("please refer to our paper or contact the authors.")
    print("=" * 80)