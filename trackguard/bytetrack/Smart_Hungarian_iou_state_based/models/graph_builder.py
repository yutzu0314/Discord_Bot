"""
Graph Builder for GBC-MOT POC
Uses centralized settings from utils.settings
Implements Equations 2-7 from the paper for graph construction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch_geometric
from torch_geometric.data import Data
import cv2
import time

class GraphBuilder:
    """
    Graph construction for GBC-MOT tracking
    Builds dynamic graphs from detection and track information
    """
    
    def __init__(self, config=None):
        """
        Initialize graph builder using centralized settings
        
        Args:
            config: Configuration object (optional, uses centralized settings if None)
        """
        from utils.settings import SETTINGS
    
        # Use centralized settings if no config provided
        if config is None:
            graph_config = SETTINGS.get_graph_config()
            self.device = torch.device(graph_config['device'])
        else:
            # For backward compatibility
            if hasattr(config, 'graph'):
                graph_config = config.graph.__dict__
                self.device = torch.device(config.graph.device if hasattr(config.graph, 'device') 
                                        else ('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                graph_config = config
                self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Graph construction parameters from config
        self.position_dim = graph_config['position_dim']
        self.size_dim = graph_config['size_dim']
        self.appearance_dim = graph_config['appearance_dim']
        self.hidden_dim = graph_config['hidden_dim']
        self.temporal_dim = graph_config['temporal_dim']
        
        # Edge computation weights (Equations 3-7)
        self.alpha = graph_config['distance_weight']
        self.beta = graph_config['temporal_weight']
        self.similarity_weight = graph_config['similarity_weight']
        self.motion_weight = graph_config['motion_weight']
        
        # Graph construction parameters
        self.max_distance = graph_config['max_distance_threshold']
        self.min_similarity = graph_config['min_similarity_threshold']
        self.max_neighbors = graph_config['max_neighbors']
        self.use_knn = graph_config['use_knn_graph']
        self.k_neighbors = graph_config['k_neighbors']
        self.epsilon = graph_config['epsilon']
        
        print(f"✓ Graph builder initialized on {self.device}")
        print(f"  Node dims: pos={self.position_dim}, size={self.size_dim}, "
            f"app={self.appearance_dim}, hidden={self.hidden_dim}")
        print(f"  Edge weights: α={self.alpha}, β={self.beta}")
        print(f"  🎯 Using centralized settings")
        
    def build_nodes(self, detections: List[Dict], features: np.ndarray, 
                   hidden_states: Optional[np.ndarray] = None,
                   temporal_weights: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Build node representations according to Equation 2
        v_i^t = (p_i^t, s_i^t, a_i^t, h_i^t, τ_i^t)
        
        Args:
            detections: List of detection dictionaries
            features: Appearance features [N, appearance_dim]
            hidden_states: Hidden states [N, hidden_dim] (optional)
            temporal_weights: Temporal weights [N, temporal_dim] (optional)
            
        Returns:
            Node feature matrix [N, total_node_dim]
        """
        num_nodes = len(detections)
        if num_nodes == 0:
            return torch.empty((0, self._get_node_dim()), device=self.device)
        
        # Extract position and size from detections
        positions = []  # p_i^t
        sizes = []      # s_i^t
        
        for det in detections:
            # Position: center coordinates
            center = det['center']
            positions.append([center[0], center[1]])
            
            # Size: width and height
            size = det['size']
            sizes.append([size[0], size[1]])
        
        positions = np.array(positions, dtype=np.float32)  # [N, 2]
        sizes = np.array(sizes, dtype=np.float32)          # [N, 2]
        
        # Normalize positions and sizes for better training
        positions = self._normalize_positions(positions)
        sizes = self._normalize_sizes(sizes)
        
        # Handle missing components
        if hidden_states is None:
            hidden_states = np.zeros((num_nodes, self.hidden_dim), dtype=np.float32)
        
        if temporal_weights is None:
            temporal_weights = np.ones((num_nodes, self.temporal_dim), dtype=np.float32)
        
        # Ensure feature dimensions match
        if features.shape[0] != num_nodes:
            raise ValueError(f"Feature count {features.shape[0]} != detection count {num_nodes}")
        
        # Concatenate all node features: [p_i^t, s_i^t, a_i^t, h_i^t, τ_i^t]
        node_features = np.concatenate([
            positions,         # [N, position_dim]
            sizes,            # [N, size_dim]  
            features,         # [N, appearance_dim]
            hidden_states,    # [N, hidden_dim]
            temporal_weights  # [N, temporal_dim]
        ], axis=1)
        
        # Convert to tensor
        node_tensor = torch.from_numpy(node_features).float().to(self.device)
        
        return node_tensor
    
    def compute_edge_weights(self, detections: List[Dict], features: np.ndarray,
                           prev_detections: Optional[List[Dict]] = None,
                           time_diff: float = 1.0) -> np.ndarray:
        """
        Compute edge weights according to Equations 3-7
        
        Args:
            detections: Current frame detections
            features: Appearance features
            prev_detections: Previous frame detections (for temporal)
            time_diff: Time difference between frames
            
        Returns:
            Edge weight matrix [N, N]
        """
        num_nodes = len(detections)
        if num_nodes == 0:
            return np.zeros((0, 0), dtype=np.float32)
        
        # Initialize components
        D_matrix = self._compute_distance_matrix(detections)      # Equation 3
        S_matrix = self._compute_similarity_matrix(features)      # Equation 4  
        T_matrix = self._compute_temporal_matrix(num_nodes, time_diff)  # Equation 5
        M_matrix = self._compute_motion_matrix(detections, prev_detections)  # Equation 6
        
        # Combine components according to Equation 7
        # e_{ij}^{t,t'} = σ(W_e Φ(v_i^t, v_j^{t'}))
        # where Φ = [D, S, T, M]
        edge_weights = (D_matrix * self.alpha + 
                       S_matrix * self.similarity_weight +
                       T_matrix * self.beta +
                       M_matrix * self.motion_weight)
        
        # Apply sigmoid activation
        edge_weights = self._sigmoid(edge_weights)
        
        return edge_weights
    
    def _compute_distance_matrix(self, detections: List[Dict]) -> np.ndarray:
        """
        Compute distance matrix according to Equation 3
        D_{ij}^{t,t'} = exp(-α ||p_i^t - p_j^{t'}||_2^2)
        """
        num_nodes = len(detections)
        positions = np.array([det['center'] for det in detections])
        
        # Compute pairwise distances
        distances = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist_sq = np.sum((positions[i] - positions[j])**2)
                    distances[i, j] = np.exp(-self.alpha * dist_sq / (self.max_distance**2))
                else:
                    distances[i, j] = 1.0  # Self-connection
        
        return distances
    
    def _compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute appearance similarity matrix according to Equation 4
        S_{ij}^{t,t'} = (a_i^t · a_j^{t'}) / max(||a_i^t||_2 ||a_j^{t'}||_2, ε)
        """
        num_nodes = features.shape[0]
        
        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, self.epsilon)
        normalized_features = features / norms
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized_features, normalized_features.T)
        
        # Ensure diagonal is 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _compute_temporal_matrix(self, num_nodes: int, time_diff: float) -> np.ndarray:
        """
        Compute temporal matrix according to Equation 5
        T_{ij}^{t,t'} = exp(-β |t - t'|)
        """
        # For within-frame graph, temporal difference is 0
        # All nodes are from same frame
        temporal_matrix = np.exp(-self.beta * time_diff)
        temporal_matrix = np.full((num_nodes, num_nodes), temporal_matrix)
        
        return temporal_matrix
    
    def _compute_motion_matrix(self, detections: List[Dict], 
                             prev_detections: Optional[List[Dict]]) -> np.ndarray:
        """
        Compute motion matrix according to Equation 6 (IoU-based)
        M_{ij}^{t,t'} = IoU(B_i^t, B_j^{t'})
        """
        num_nodes = len(detections)
        
        if prev_detections is None:
            # Within-frame: use overlap between current detections
            motion_matrix = np.zeros((num_nodes, num_nodes))
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        iou = self._compute_bbox_iou(detections[i]['bbox'], 
                                                   detections[j]['bbox'])
                        motion_matrix[i, j] = iou
                    else:
                        motion_matrix[i, j] = 1.0
        else:
            # Cross-frame: compute IoU between current and previous
            # For now, use spatial overlap as proxy
            motion_matrix = np.ones((num_nodes, num_nodes)) * 0.5
        
        return motion_matrix
    
    def _compute_bbox_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def construct_graph(self, detections: List[Dict], features: np.ndarray,
                       hidden_states: Optional[np.ndarray] = None,
                       temporal_weights: Optional[np.ndarray] = None,
                       prev_detections: Optional[List[Dict]] = None) -> Data:
        """
        Construct complete graph according to Equations 2-7
        
        Args:
            detections: Current frame detections
            features: Appearance features
            hidden_states: Hidden states (optional)
            temporal_weights: Temporal weights (optional)
            prev_detections: Previous frame detections (optional)
            
        Returns:
            PyTorch Geometric Data object
        """
        if len(detections) == 0:
            return self._create_empty_graph()
        
        # Build nodes (Equation 2)
        node_features = self.build_nodes(detections, features, hidden_states, temporal_weights)
        
        # Compute edge weights (Equations 3-7)
        edge_weights = self.compute_edge_weights(detections, features, prev_detections)
        
        # Build edge indices and attributes
        edge_index, edge_attr = self._build_edges(edge_weights)
        
        # Create PyTorch Geometric graph
        graph = Data(
            x=node_features,           # Node features [N, node_dim]
            edge_index=edge_index,     # Edge connectivity [2, E]
            edge_attr=edge_attr,       # Edge weights [E, 1]
            num_nodes=len(detections)
        )
        
        return graph
    
    def _build_edges(self, edge_weights: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge indices and attributes from weight matrix
        
        Args:
            edge_weights: Edge weight matrix [N, N]
            
        Returns:
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge attributes [E, 1]
        """
        num_nodes = edge_weights.shape[0]
        
        if self.use_knn:
            # Use K-nearest neighbors for sparse graph
            edge_list = []
            edge_weights_list = []
            
            for i in range(num_nodes):
                # Get top-k neighbors (excluding self)
                weights_i = edge_weights[i].copy()
                weights_i[i] = -1  # Exclude self
                
                top_k_indices = np.argsort(weights_i)[-self.k_neighbors:]
                
                for j in top_k_indices:
                    if weights_i[j] > self.min_similarity:
                        edge_list.append([i, j])
                        edge_weights_list.append(edge_weights[i, j])
        else:
            # Use threshold-based graph
            edge_list = []
            edge_weights_list = []
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and edge_weights[i, j] > self.min_similarity:
                        # Apply distance threshold
                        if self._check_distance_threshold(i, j, edge_weights):
                            edge_list.append([i, j])
                            edge_weights_list.append(edge_weights[i, j])
        
        # Convert to tensors
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights_list, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        return edge_index.to(self.device), edge_attr.to(self.device)
    
    def _check_distance_threshold(self, i: int, j: int, edge_weights: np.ndarray) -> bool:
        """Check if edge satisfies distance threshold"""
        # For now, use edge weight as proxy
        return edge_weights[i, j] > self.min_similarity
    
    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to [0, 1] range"""
        if len(positions) == 0:
            return positions
        
        # Simple min-max normalization
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        range_pos = max_pos - min_pos
        range_pos = np.maximum(range_pos, self.epsilon)
        
        normalized = (positions - min_pos) / range_pos
        return normalized
    
    def _normalize_sizes(self, sizes: np.ndarray) -> np.ndarray:
        """Normalize sizes"""
        if len(sizes) == 0:
            return sizes
        
        # Log normalization for sizes
        normalized = np.log(sizes + 1.0)
        return normalized
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _get_node_dim(self) -> int:
        """Get total node feature dimension"""
        return (self.position_dim + self.size_dim + self.appearance_dim + 
                self.hidden_dim + self.temporal_dim)
    
    def _create_empty_graph(self) -> Data:
        """Create empty graph for edge cases"""
        return Data(
            x=torch.empty((0, self._get_node_dim()), device=self.device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
            edge_attr=torch.empty((0, 1), device=self.device),
            num_nodes=0
        )
    
    def visualize_graph(self, graph: Data, detections: List[Dict], 
                       image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize graph structure for debugging
        
        Args:
            graph: PyTorch Geometric graph
            detections: Detection list for positioning
            image: Background image (optional)
            
        Returns:
            Visualization image
        """
        if image is not None:
            vis_image = image.copy()
        else:
            vis_image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        if graph.num_nodes == 0:
            cv2.putText(vis_image, "Empty Graph", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return vis_image
        
        # Draw nodes
        node_positions = []
        for i, det in enumerate(detections):
            center = det['center']
            node_positions.append(center)
            
            # Draw node
            cv2.circle(vis_image, (int(center[0]), int(center[1])), 
                      8, (0, 255, 0), -1)
            
            # Draw node ID
            cv2.putText(vis_image, str(i), 
                       (int(center[0] + 10), int(center[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw edges
        edge_index = graph.edge_index.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy()
        
        for e in range(edge_index.shape[1]):
            i, j = edge_index[0, e], edge_index[1, e]
            weight = edge_attr[e, 0]
            
            pos_i = node_positions[i]
            pos_j = node_positions[j]
            
            # Color based on weight
            color_intensity = int(255 * weight)
            color = (color_intensity, color_intensity, 0)
            
            cv2.line(vis_image, 
                    (int(pos_i[0]), int(pos_i[1])),
                    (int(pos_j[0]), int(pos_j[1])),
                    color, 2)
        
        # Add info
        info_text = f"Nodes: {graph.num_nodes}, Edges: {edge_index.shape[1]}"
        cv2.putText(vis_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def get_graph_stats(self, graph: Data) -> Dict:
        """Get graph statistics"""
        edge_index = graph.edge_index.cpu().numpy()
        
        stats = {
            'num_nodes': graph.num_nodes,
            'num_edges': edge_index.shape[1] if edge_index.size > 0 else 0,
            'node_feature_dim': graph.x.shape[1] if graph.x.size(0) > 0 else 0,
            'avg_degree': edge_index.shape[1] / graph.num_nodes if graph.num_nodes > 0 else 0,
            'is_directed': True  # Our graph is directed
        }
        
        if edge_index.size > 0:
            edge_weights = graph.edge_attr.cpu().numpy()
            stats.update({
                'avg_edge_weight': np.mean(edge_weights),
                'min_edge_weight': np.min(edge_weights), 
                'max_edge_weight': np.max(edge_weights)
            })
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test graph builder
    from utils.config import get_config
    
    config = get_config()
    builder = GraphBuilder(config)
    
    # Create dummy data
    dummy_detections = [
        {'center': [100, 200], 'size': [50, 100], 'bbox': [75, 150, 125, 250]},
        {'center': [200, 300], 'size': [60, 120], 'bbox': [170, 240, 230, 360]},
        {'center': [150, 250], 'size': [55, 110], 'bbox': [122, 195, 178, 305]}
    ]
    
    dummy_features = np.random.randn(3, 128).astype(np.float32)
    
    print("Testing graph construction...")
    
    # Build graph
    graph = builder.construct_graph(dummy_detections, dummy_features)
    
    # Print stats
    stats = builder.get_graph_stats(graph)
    print("Graph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test visualization
    vis_image = builder.visualize_graph(graph, dummy_detections)
    print(f"Visualization image shape: {vis_image.shape}")