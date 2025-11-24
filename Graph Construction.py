"""
Multi-Topology Graph Construction Module for PERMA-GNN-Transformer

This module implements four types of student relationship graphs:
1. Cosine Similarity Graph - captures angular relationships in feature space
2. Euclidean Distance Graph - reflects overall feature differences (k-NN)
3. Learning Style Graph - models personalized learning patterns
4. PERMA-Weighted Graph - theory-driven psychological relationship modeling

Paper Reference: Section 3.2.1 Multi-source Feature Input and Student Relationship Graph Construction
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform, cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Tuple, Optional, Dict
import networkx as nx


class GraphTopologyConfig:
    """Configuration for graph construction parameters (from paper Section 4.1.3)"""

    # Cosine similarity threshold (paper: 0.3)
    COSINE_THRESHOLD = 0.3

    # Euclidean distance k-NN parameter
    EUCLIDEAN_K = 10

    # Learning style clustering parameters
    N_LEARNING_STYLES = 4  # Visual, Auditory, Reading/Writing, Kinesthetic

    # PERMA-weighted graph parameters
    PERMA_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights as baseline
    PERMA_THRESHOLD = 0.3

    # Edge weight normalization
    USE_SYMMETRIC_NORM = True


class GraphConstructor:
    """
    Builds four types of student relationship graphs

    As described in paper Section 3.2.1, these graphs capture different aspects
    of student relationships and are processed in parallel by multi-topology GNN.
    """

    def __init__(self, config: Optional[GraphTopologyConfig] = None):
        self.config = config or GraphTopologyConfig()

    def construct_all_topologies(
            self,
            features: torch.Tensor,
            perma_features: torch.Tensor,
            learning_styles: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Construct all four graph topologies

        Args:
            features: Student features [n_students, n_features]
            perma_features: PERMA dimension embeddings [n_students, 5, embed_dim]
            learning_styles: Learning style indicators [n_students] (optional)

        Returns:
            List of 4 edge_index tensors, one for each topology
        """
        n_students = features.shape[0]

        # Convert to numpy for distance computations if needed
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features

        # 1. Cosine Similarity Graph
        print("Constructing Cosine Similarity Graph...")
        cosine_edge_index = self.construct_cosine_graph(features, self.config.COSINE_THRESHOLD)
        print(f"  → {cosine_edge_index.shape[1]} edges")

        # 2. Euclidean Distance Graph (k-NN)
        print("Constructing Euclidean Distance k-NN Graph...")
        euclidean_edge_index = self.construct_euclidean_graph(features, self.config.EUCLIDEAN_K)
        print(f"  → {euclidean_edge_index.shape[1]} edges")

        # 3. Learning Style Graph
        print("Constructing Learning Style Graph...")
        if learning_styles is None:
            learning_styles = self._infer_learning_styles(features)
        learning_edge_index = self.construct_learning_style_graph(learning_styles)
        print(f"  → {learning_edge_index.shape[1]} edges")

        # 4. PERMA-Weighted Graph
        print("Constructing PERMA-Weighted Graph...")
        perma_edge_index = self.construct_perma_weighted_graph(
            perma_features,
            self.config.PERMA_WEIGHTS,
            self.config.PERMA_THRESHOLD
        )
        print(f"  → {perma_edge_index.shape[1]} edges")

        edge_index_list = [
            cosine_edge_index,
            euclidean_edge_index,
            learning_edge_index,
            perma_edge_index
        ]

        # Ensure all are on the same device
        device = features.device if torch.is_tensor(features) else torch.device('cpu')
        edge_index_list = [ei.to(device) for ei in edge_index_list]

        return edge_index_list

    def construct_cosine_graph(
            self,
            features: torch.Tensor,
            threshold: float = 0.3
    ) -> torch.Tensor:
        """
        Construct graph based on cosine similarity

        Formula (from paper Section 3.2.1):
            a_ij^(cos) = 1 if cos_sim(x_i, x_j) > threshold, else 0

        Args:
            features: Student features [n_students, n_features]
            threshold: Similarity threshold (default 0.3 from paper)

        Returns:
            edge_index: [2, n_edges] - COO format edge list
        """
        if torch.is_tensor(features):
            # Normalize features for cosine similarity
            features_norm = F.normalize(features, p=2, dim=1)
            # Compute cosine similarity matrix
            similarity_matrix = torch.mm(features_norm, features_norm.t())
            similarity_matrix = similarity_matrix.cpu().numpy()
        else:
            # Use sklearn for numpy arrays
            similarity_matrix = cosine_similarity(features)

        # Remove self-loops
        np.fill_diagonal(similarity_matrix, 0)

        # Apply threshold
        adjacency = (similarity_matrix > threshold).astype(float)

        # Convert to edge_index format (COO)
        edge_index = self._adjacency_to_edge_index(adjacency)

        return edge_index

    def construct_euclidean_graph(
            self,
            features: torch.Tensor,
            k: int = 10
    ) -> torch.Tensor:
        """
        Construct k-NN graph based on Euclidean distance

        Formula (from paper Section 3.2.1):
            a_ij^(euc) = 1 if j ∈ k-NN(i), else 0

        Each node connects to its k nearest neighbors by Euclidean distance.

        Args:
            features: Student features [n_students, n_features]
            k: Number of nearest neighbors

        Returns:
            edge_index: [2, n_edges] - COO format edge list
        """
        if torch.is_tensor(features):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        n_students = features_np.shape[0]

        # Compute pairwise Euclidean distances
        dist_matrix = euclidean_distances(features_np)

        # Remove self-loops by setting diagonal to inf
        np.fill_diagonal(dist_matrix, np.inf)

        # Find k nearest neighbors for each student
        adjacency = np.zeros((n_students, n_students))

        for i in range(n_students):
            # Get indices of k nearest neighbors
            k_nearest_indices = np.argpartition(dist_matrix[i], k)[:k]
            adjacency[i, k_nearest_indices] = 1

        # Make symmetric (if i is neighbor of j, j is neighbor of i)
        adjacency = np.maximum(adjacency, adjacency.T)

        # Convert to edge_index
        edge_index = self._adjacency_to_edge_index(adjacency)

        return edge_index

    def construct_learning_style_graph(
            self,
            learning_styles: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct graph based on learning style similarity

        Students with the same learning style are connected.
        Learning styles: Visual, Auditory, Reading/Writing, Kinesthetic

        Formula (from paper Section 3.2.1):
            a_ij^(style) = 1 if style(i) == style(j), else 0

        Args:
            learning_styles: Learning style labels [n_students]
                            Values in {0, 1, 2, 3} for 4 learning styles

        Returns:
            edge_index: [2, n_edges] - COO format edge list
        """
        if torch.is_tensor(learning_styles):
            learning_styles_np = learning_styles.cpu().numpy()
        else:
            learning_styles_np = learning_styles

        n_students = len(learning_styles_np)
        adjacency = np.zeros((n_students, n_students))

        # Connect students with the same learning style
        for style in np.unique(learning_styles_np):
            style_indices = np.where(learning_styles_np == style)[0]
            # Create complete subgraph for this learning style
            for i in style_indices:
                for j in style_indices:
                    if i != j:
                        adjacency[i, j] = 1

        # Convert to edge_index
        edge_index = self._adjacency_to_edge_index(adjacency)

        return edge_index

    def construct_perma_weighted_graph(
            self,
            perma_features: torch.Tensor,
            weights: Optional[List[float]] = None,
            threshold: float = 0.3
    ) -> torch.Tensor:
        """
        Construct theory-driven PERMA-weighted graph

        Formula (from paper Section 3.2.1):
            a_ij^(PERMA) = Σ(p=1 to 5) w_p · sim_p(x_i^(p), x_j^(p))

        where w_p are theory-based weights for each PERMA dimension,
        and sim_p is the cosine similarity in dimension p.

        Args:
            perma_features: PERMA embeddings [n_students, 5, embed_dim]
            weights: Weight for each PERMA dimension (default: equal weights)
            threshold: Similarity threshold

        Returns:
            edge_index: [2, n_edges] - COO format edge list
        """
        if weights is None:
            weights = self.config.PERMA_WEIGHTS

        weights = torch.tensor(weights, dtype=torch.float32)
        if torch.is_tensor(perma_features):
            weights = weights.to(perma_features.device)

        n_students = perma_features.shape[0]
        n_dims = perma_features.shape[1]  # Should be 5 for PERMA

        # Initialize weighted similarity matrix
        weighted_similarity = torch.zeros(n_students, n_students)
        if torch.is_tensor(perma_features):
            weighted_similarity = weighted_similarity.to(perma_features.device)

        # Compute similarity for each PERMA dimension
        for p in range(n_dims):
            # Extract features for dimension p: [n_students, embed_dim]
            dim_features = perma_features[:, p, :]

            # Normalize for cosine similarity
            dim_features_norm = F.normalize(dim_features, p=2, dim=1)

            # Compute cosine similarity matrix
            dim_similarity = torch.mm(dim_features_norm, dim_features_norm.t())

            # Add weighted contribution
            weighted_similarity += weights[p] * dim_similarity

        # Average by sum of weights
        weighted_similarity = weighted_similarity / weights.sum()

        # Remove self-loops
        weighted_similarity.fill_diagonal_(0)

        # Apply threshold
        if torch.is_tensor(weighted_similarity):
            adjacency = (weighted_similarity > threshold).float()
            adjacency_np = adjacency.cpu().numpy()
        else:
            adjacency_np = (weighted_similarity > threshold).astype(float)

        # Convert to edge_index
        edge_index = self._adjacency_to_edge_index(adjacency_np)

        return edge_index

    def _adjacency_to_edge_index(self, adjacency: np.ndarray) -> torch.Tensor:
        """
        Convert adjacency matrix to edge_index (COO format)

        Args:
            adjacency: [n, n] adjacency matrix

        Returns:
            edge_index: [2, n_edges] edge list
        """
        # Find non-zero entries
        edges = np.array(np.where(adjacency > 0))
        edge_index = torch.LongTensor(edges)

        return edge_index

    def _infer_learning_styles(self, features: torch.Tensor) -> torch.Tensor:
        """
        Infer learning styles from student features using clustering

        This is a simplified heuristic. In practice, learning styles would be:
        1. Explicitly provided in the dataset
        2. Derived from educational psychology assessments
        3. Inferred from behavioral patterns

        Args:
            features: Student features [n_students, n_features]

        Returns:
            learning_styles: Cluster labels [n_students]
        """
        from sklearn.cluster import KMeans

        if torch.is_tensor(features):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        # Use K-means clustering to infer 4 learning style groups
        kmeans = KMeans(n_clusters=self.config.N_LEARNING_STYLES, random_state=42)
        learning_styles = kmeans.fit_predict(features_np)

        return torch.LongTensor(learning_styles)

    def compute_edge_weights(
            self,
            features: torch.Tensor,
            edge_index: torch.Tensor,
            similarity_type: str = 'cosine'
    ) -> torch.Tensor:
        """
        Compute edge weights based on feature similarity

        Args:
            features: Node features [n_nodes, n_features]
            edge_index: Edge list [2, n_edges]
            similarity_type: 'cosine' or 'euclidean'

        Returns:
            edge_weights: [n_edges]
        """
        n_edges = edge_index.shape[1]
        edge_weights = torch.zeros(n_edges)

        if torch.is_tensor(features):
            edge_weights = edge_weights.to(features.device)

        for i in range(n_edges):
            src, dst = edge_index[0, i], edge_index[1, i]

            if similarity_type == 'cosine':
                # Cosine similarity
                sim = F.cosine_similarity(
                    features[src].unsqueeze(0),
                    features[dst].unsqueeze(0)
                ).item()
            else:  # euclidean
                # Inverse of Euclidean distance (normalized)
                dist = torch.dist(features[src], features[dst]).item()
                sim = 1.0 / (1.0 + dist)

            edge_weights[i] = sim

        return edge_weights

    def normalize_adjacency(
            self,
            edge_index: torch.Tensor,
            n_nodes: int,
            normalization: str = 'symmetric'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize adjacency matrix for GCN

        Symmetric normalization (from paper Section 4.1.3):
            Â = D^(-1/2) A D^(-1/2)

        Args:
            edge_index: [2, n_edges]
            n_nodes: Number of nodes
            normalization: 'symmetric' or 'row'

        Returns:
            edge_index: Normalized edge list
            edge_weight: Edge weights after normalization
        """
        # Compute degree matrix
        row, col = edge_index
        deg = torch.bincount(row, minlength=n_nodes).float()

        if normalization == 'symmetric':
            # D^(-1/2)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            # Edge weights: D^(-1/2) * A * D^(-1/2)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        elif normalization == 'row':
            # D^(-1)
            deg_inv = 1.0 / deg
            deg_inv[deg_inv == float('inf')] = 0

            # Edge weights: D^(-1) * A
            edge_weight = deg_inv[row]

        else:
            edge_weight = torch.ones(edge_index.shape[1])

        return edge_index, edge_weight


class GraphStatistics:
    """Utility class to compute graph statistics for analysis"""

    @staticmethod
    def compute_graph_stats(edge_index: torch.Tensor, n_nodes: int) -> Dict:
        """
        Compute basic graph statistics

        Args:
            edge_index: [2, n_edges]
            n_nodes: Number of nodes

        Returns:
            Dictionary with statistics
        """
        n_edges = edge_index.shape[1]

        # Compute degrees
        degrees = torch.bincount(edge_index[0], minlength=n_nodes)

        # Compute density
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0

        stats = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': degrees.float().mean().item(),
            'max_degree': degrees.max().item(),
            'min_degree': degrees.min().item(),
            'density': density,
            'is_directed': not GraphStatistics._is_symmetric(edge_index)
        }

        return stats

    @staticmethod
    def _is_symmetric(edge_index: torch.Tensor) -> bool:
        """Check if graph is symmetric (undirected)"""
        # Create set of edges
        edges_set = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edges_set.add((src, dst))

        # Check if reverse edge exists for each edge
        for src, dst in edges_set:
            if (dst, src) not in edges_set:
                return False

        return True

    @staticmethod
    def print_graph_comparison(edge_index_list: List[torch.Tensor], n_nodes: int):
        """
        Print comparison of graph topologies

        Args:
            edge_index_list: List of 4 edge_index tensors
            n_nodes: Number of nodes
        """
        topology_names = [
            "Cosine Similarity",
            "Euclidean Distance",
            "Learning Style",
            "PERMA-Weighted"
        ]

        print("\n" + "=" * 80)
        print("Multi-Topology Graph Statistics")
        print("=" * 80)

        for i, (name, edge_index) in enumerate(zip(topology_names, edge_index_list)):
            stats = GraphStatistics.compute_graph_stats(edge_index, n_nodes)

            print(f"\n{i + 1}. {name} Graph:")
            print(f"   Edges: {stats['n_edges']}")
            print(f"   Avg Degree: {stats['avg_degree']:.2f}")
            print(f"   Density: {stats['density']:.4f}")
            print(f"   Degree Range: [{stats['min_degree']}, {stats['max_degree']}]")

        print("\n" + "=" * 80)


def create_graph_batch(
        edge_index_list: List[torch.Tensor],
        batch_indices: torch.Tensor,
        n_nodes_total: int
) -> List[torch.Tensor]:
    """
    Create batched graphs for mini-batch training

    Args:
        edge_index_list: List of 4 edge_index tensors for full graph
        batch_indices: Node indices in current batch [batch_size]
        n_nodes_total: Total number of nodes in full graph

    Returns:
        List of 4 edge_index tensors for the batch subgraph
    """
    batch_size = len(batch_indices)
    batch_set = set(batch_indices.cpu().numpy())

    batched_edge_list = []

    for edge_index in edge_index_list:
        # Filter edges: both endpoints must be in batch
        mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in batch_set and dst in batch_set:
                mask[i] = True

        # Extract subgraph
        batch_edge_index = edge_index[:, mask]

        # Remap node indices to batch-local indices
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(batch_indices.cpu().numpy())}

        remapped_edges = torch.zeros_like(batch_edge_index)
        for i in range(batch_edge_index.shape[1]):
            src, dst = batch_edge_index[0, i].item(), batch_edge_index[1, i].item()
            remapped_edges[0, i] = node_map[src]
            remapped_edges[1, i] = node_map[dst]

        batched_edge_list.append(remapped_edges)

    return batched_edge_list


if __name__ == "__main__":
    """
    Example usage demonstrating multi-topology graph construction
    """

    print("PERMA-GNN-Transformer: Multi-Topology Graph Construction")
    print("=" * 80)

    # Simulate student data
    n_students = 100
    n_features = 23  # Lifestyle dataset
    embed_dim = 128

    # Generate synthetic data
    torch.manual_seed(42)
    features = torch.randn(n_students, n_features)
    perma_features = torch.randn(n_students, 5, embed_dim)

    print(f"\nStudent data: {n_students} students, {n_features} features")

    # Construct graphs
    constructor = GraphConstructor()
    edge_index_list = constructor.construct_all_topologies(
        features=features,
        perma_features=perma_features
    )

    # Print statistics
    GraphStatistics.print_graph_comparison(edge_index_list, n_students)

    # Demonstrate edge weight computation
    print("\n" + "=" * 80)
    print("Edge Weight Example (Cosine Similarity Graph)")
    print("=" * 80)

    cosine_edge_index = edge_index_list[0]
    edge_weights = constructor.compute_edge_weights(
        features, cosine_edge_index, similarity_type='cosine'
    )

    print(f"\nFirst 10 edge weights:")
    for i in range(min(10, edge_weights.shape[0])):
        src, dst = cosine_edge_index[0, i].item(), cosine_edge_index[1, i].item()
        weight = edge_weights[i].item()
        print(f"  Edge ({src} → {dst}): weight = {weight:.4f}")

    print("\n" + "=" * 80)
    print("Graph Construction Complete!")
    print("=" * 80)