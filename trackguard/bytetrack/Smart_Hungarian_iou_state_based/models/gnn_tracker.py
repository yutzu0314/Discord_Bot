"""
GNN Tracker for GBC-MOT POC
Uses centralized settings from utils.settings
Implements GAT-based message passing according to Equations 8-9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
import time

class GATTracker(nn.Module):
    """
    Graph Attention Network based tracker
    Implements Equations 8-9 for message passing and node updates
    """
    
    def __init__(self, config=None):
        """
        Initialize GAT tracker using centralized settings
        
        Args:
            config: Configuration object (optional, uses centralized settings if None)
        """
        super(GATTracker, self).__init__()
    
        from utils.settings import SETTINGS
        
        # Use centralized settings if no config provided
        if config is None:
            gnn_config = SETTINGS.get_gnn_config()
            graph_config = SETTINGS.get_graph_config()
            self.device = torch.device(gnn_config['device'])
        else:
            # For backward compatibility
            if hasattr(config, 'gnn'):
                gnn_config = config.gnn.__dict__
                self.device = torch.device(config.gnn.device if hasattr(config.gnn, 'device') 
                                        else ('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                gnn_config = config
                self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            
            if hasattr(config, 'graph'):
                graph_config = config.graph.__dict__
            else:
                # Use centralized settings for graph config if not provided
                graph_config = SETTINGS.get_graph_config()
        
        # Architecture parameters
        self.num_layers = gnn_config['num_layers']
        self.hidden_dim = gnn_config['hidden_dim']
        self.num_heads = gnn_config['num_heads']
        self.dropout = gnn_config['dropout']
        self.prediction_dim = gnn_config['prediction_dim']
        
        # Input dimensions from graph builder
        self.position_dim = graph_config['position_dim']
        self.size_dim = graph_config['size_dim']
        self.appearance_dim = graph_config['appearance_dim']
        self.graph_hidden_dim = graph_config['hidden_dim']
        self.temporal_dim = graph_config['temporal_dim']
        
        self.input_dim = (self.position_dim + self.size_dim + self.appearance_dim + 
                        self.graph_hidden_dim + self.temporal_dim)
        
        # Build network layers
        self._build_layers()
        
        # Move to device
        self.to(self.device)
        
        print(f"✓ GAT tracker initialized on {self.device}")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Heads: {self.num_heads}")
        print(f"  Prediction dim: {self.prediction_dim}")
        print(f"  🎯 Using centralized settings")
    
    def _build_layers(self):
        """Build GAT layers according to Equations 8-9"""
        
        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # GAT layers for message passing (Equations 8-9)
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                dropout=self.dropout,
                edge_dim=1,  # Edge attributes dimension
                concat=True
            )
        )
        
        # Additional GAT layers
        for _ in range(1, self.num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.num_heads,
                    heads=self.num_heads,
                    dropout=self.dropout,
                    edge_dim=1,
                    concat=True
                )
            )
        
        # Prediction heads (Equations 10-11 from paper)
        self.position_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.position_dim),
            nn.Tanh()  # Bounded position changes
        )
        
        self.size_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.size_dim),
            nn.ReLU()  # Non-negative size changes
        )
        
        # Combined prediction for tracking
        self.combined_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.prediction_dim)  # [dx, dy, dw, dh]
        )
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Activation function
        # Get activation from centralized config
        from utils.settings import SETTINGS
        gnn_config = SETTINGS.get_gnn_config()
        activation_type = gnn_config.get('activation', 'relu')

        if activation_type == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation_type == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
            
    
    def forward(self, graph: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing Equations 8-9
        
        Args:
            graph: PyTorch Geometric graph with node features and edges
            
        Returns:
            Dictionary with predictions and hidden states
        """
        if graph.num_nodes == 0:
            return self._empty_predictions()
        
        # Input projection
        x = self.input_projection(graph.x)  # [N, hidden_dim]
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        # Store initial hidden state
        h_initial = x.clone()
        
        # Message passing through GAT layers (Equations 8-9)
        hidden_states = []
        
        for layer_idx, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # GAT message passing
            # m_{ij}^{t,(l)} = φ_m^{(l)}(h_i^{t,(l)}, h_j^{t,(l)}, e_{ij}^t)  [Eq 8]
            # h_i^{t,(l+1)} = φ_h^{(l)}(h_i^{t,(l)}, AGGREGATE(...))         [Eq 9]
            
            x_new = gat_layer(x, edge_index, edge_attr)
            
            # Apply activation and layer norm
            x_new = self.activation(x_new)
            x_new = layer_norm(x_new)
            
            # Residual connection for better training
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            hidden_states.append(x.clone())
        
        # Final hidden state after L layers
        h_final = x  # h_i^{t,(L)}
        
        # Predictions using final hidden state
        predictions = self._compute_predictions(h_final)
        
        return {
            'hidden_states': hidden_states,
            'final_hidden': h_final,
            'initial_hidden': h_initial,
            'position_prediction': predictions['position'],
            'size_prediction': predictions['size'],
            'combined_prediction': predictions['combined']
        }
    
    def _compute_predictions(self, h_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute predictions from final hidden state
        
        Args:
            h_final: Final hidden state [N, hidden_dim]
            
        Returns:
            Dictionary with different prediction types
        """
        # Position prediction (Equation 10)
        # p_i^{t+1} = σ_p(W_p h_i^{t,(L)} + b_p)
        position_pred = self.position_predictor(h_final)  # [N, 2] - [dx, dy]
        
        # Size prediction (Equation 11)  
        # s_i^{t+1} = ReLU(W_s h_i^{t,(L)} + b_s)
        size_pred = self.size_predictor(h_final)  # [N, 2] - [dw, dh]
        
        # Combined prediction for tracking
        combined_pred = self.combined_predictor(h_final)  # [N, 4] - [dx, dy, dw, dh]
        
        return {
            'position': position_pred,
            'size': size_pred,
            'combined': combined_pred
        }
    
    def predict_next_frame(self, graph: Data, current_detections: List[Dict]) -> Dict:
        """
        Predict next frame positions and sizes
        
        Args:
            graph: Current frame graph
            current_detections: Current frame detections
            
        Returns:
            Predicted detections for next frame
        """
        if graph.num_nodes == 0:
            return {'predicted_detections': [], 'confidence': []}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(graph)
            
            # Get predictions
            position_pred = outputs['position_prediction'].cpu().numpy()  # [N, 2]
            size_pred = outputs['size_prediction'].cpu().numpy()         # [N, 2]
            combined_pred = outputs['combined_prediction'].cpu().numpy()  # [N, 4]
        
        # Apply predictions to current detections
        predicted_detections = []
        confidence_scores = []
        
        for i, det in enumerate(current_detections):
            if i >= len(combined_pred):
                break
            
            # Current state
            current_center = det['center']
            current_size = det['size']
            
            # Predicted changes
            dx, dy, dw, dh = combined_pred[i]
            
            # Apply predictions
            predicted_center = [
                current_center[0] + dx * 100,  # Scale factor for pixel coordinates
                current_center[1] + dy * 100
            ]
            
            predicted_size = [
                max(10, current_size[0] + dw * 50),  # Minimum size constraint
                max(20, current_size[1] + dh * 50)
            ]
            
            # Convert to bbox format
            pred_x1 = predicted_center[0] - predicted_size[0] / 2
            pred_y1 = predicted_center[1] - predicted_size[1] / 2
            pred_x2 = predicted_center[0] + predicted_size[0] / 2
            pred_y2 = predicted_center[1] + predicted_size[1] / 2
            
            predicted_det = {
                'center': predicted_center,
                'size': predicted_size,
                'bbox': [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)],
                'confidence': det.get('confidence', 0.8),  # Maintain or default confidence
                'predicted': True
            }
            
            predicted_detections.append(predicted_det)
            
            # Simple confidence based on prediction magnitude
            pred_magnitude = np.sqrt(dx*dx + dy*dy + dw*dw + dh*dh)
            confidence = max(0.1, 1.0 - pred_magnitude * 0.1)
            confidence_scores.append(confidence)
        
        return {
            'predicted_detections': predicted_detections,
            'confidence': confidence_scores,
            'raw_predictions': {
                'position': position_pred,
                'size': size_pred,
                'combined': combined_pred
            }
        }
    
    def _empty_predictions(self) -> Dict[str, torch.Tensor]:
        """Return empty predictions for empty graphs"""
        empty_tensor = torch.empty((0, self.hidden_dim), device=self.device)
        
        return {
            'hidden_states': [],
            'final_hidden': empty_tensor,
            'initial_hidden': empty_tensor,
            'position_prediction': torch.empty((0, self.position_dim), device=self.device),
            'size_prediction': torch.empty((0, self.size_dim), device=self.device),
            'combined_prediction': torch.empty((0, self.prediction_dim), device=self.device)
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor], 
                    weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss (simplified version for POC)
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            weights: Loss component weights
            
        Returns:
            Dictionary with loss components
        """
        if weights is None:
            weights = {'position': 1.0, 'size': 1.0, 'combined': 1.0}
        
        losses = {}
        total_loss = 0
        
        # Position loss
        if 'position_target' in targets:
            pos_loss = F.mse_loss(predictions['position_prediction'], 
                                targets['position_target'])
            losses['position_loss'] = pos_loss
            total_loss += weights['position'] * pos_loss
        
        # Size loss
        if 'size_target' in targets:
            size_loss = F.mse_loss(predictions['size_prediction'], 
                                 targets['size_target'])
            losses['size_loss'] = size_loss
            total_loss += weights['size'] * size_loss
        
        # Combined loss
        if 'combined_target' in targets:
            combined_loss = F.mse_loss(predictions['combined_prediction'], 
                                     targets['combined_target'])
            losses['combined_loss'] = combined_loss
            total_loss += weights['combined'] * combined_loss
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'GAT',
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'input_dim': self.input_dim,
            'prediction_dim': self.prediction_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
    
    def visualize_predictions(self, image: np.ndarray, 
                            current_detections: List[Dict],
                            predicted_detections: List[Dict]) -> np.ndarray:
        """
        Visualize current detections and predictions
        
        Args:
            image: Background image
            current_detections: Current frame detections
            predicted_detections: Predicted next frame detections
            
        Returns:
            Visualization image
        """
        import cv2
        
        vis_image = image.copy()
        
        # Draw current detections (green)
        for i, det in enumerate(current_detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"C{i}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw predicted detections (blue)
        for i, pred_det in enumerate(predicted_detections):
            bbox = pred_det['bbox']
            x1, y1, x2, y2 = bbox
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_image, f"P{i}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw motion vector
            if i < len(current_detections):
                curr_center = current_detections[i]['center']
                pred_center = pred_det['center']
                
                cv2.arrowedLine(vis_image,
                              (int(curr_center[0]), int(curr_center[1])),
                              (int(pred_center[0]), int(pred_center[1])),
                              (0, 255, 255), 2)
        
        # Add legend
        cv2.putText(vis_image, "Green: Current", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, "Blue: Predicted", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_image, "Yellow: Motion", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_image


# Alternative GCN implementation for comparison
class GCNTracker(nn.Module):
    """
    Simple GCN-based tracker as fallback
    """
    
    def __init__(self, config):
        super(GCNTracker, self).__init__()
        
        self.config = config
        self.device = torch.device(config.gnn.device if hasattr(config.gnn, 'device') 
                                 else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Similar structure to GAT but with GCN layers
        self.input_dim = (config.graph.position_dim + config.graph.size_dim + 
                         config.graph.appearance_dim + config.graph.hidden_dim + 
                         config.graph.temporal_dim)
        self.hidden_dim = config.gnn.hidden_dim
        self.num_layers = config.gnn.num_layers
        
        # Build GCN layers
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        self.gcn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        
        # Prediction heads (same as GAT)
        self.combined_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, config.gnn.prediction_dim)
        )
        
        self.to(self.device)
    
    def forward(self, graph: Data) -> Dict[str, torch.Tensor]:
        """Simple GCN forward pass"""
        if graph.num_nodes == 0:
            return self._empty_predictions()
        
        x = self.input_projection(graph.x)
        edge_index = graph.edge_index
        
        for gcn_layer in self.gcn_layers:
            x = F.relu(gcn_layer(x, edge_index))
        
        combined_pred = self.combined_predictor(x)
        
        return {
            'final_hidden': x,
            'combined_prediction': combined_pred,
            'position_prediction': combined_pred[:, :2],
            'size_prediction': combined_pred[:, 2:4] if combined_pred.shape[1] >= 4 else combined_pred[:, :2]
        }
    
    def _empty_predictions(self):
        """Empty predictions for GCN"""
        return {
            'final_hidden': torch.empty((0, self.hidden_dim), device=self.device),
            'combined_prediction': torch.empty((0, 4), device=self.device),
            'position_prediction': torch.empty((0, 2), device=self.device),
            'size_prediction': torch.empty((0, 2), device=self.device)
        }


# Factory function to create tracker
def create_tracker(config) -> nn.Module:
    """
    Factory function to create appropriate tracker
    
    Args:
        config: Configuration object
        
    Returns:
        Tracker model (GAT or GCN)
    """
    if config.gnn.model_type.lower() == "gat":
        return GATTracker(config)
    elif config.gnn.model_type.lower() == "gcn":
        return GCNTracker(config)
    else:
        print(f"Unknown model type: {config.gnn.model_type}, defaulting to GAT")
        return GATTracker(config)


# Example usage and testing
if __name__ == "__main__":
    # Test GAT tracker
    from utils.config import get_config
    import torch_geometric
    
    config = get_config()
    tracker = create_tracker(config)
    
    # Print model info
    info = tracker.get_model_info()
    print("GNN Tracker Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create dummy graph
    num_nodes = 5
    node_features = torch.randn(num_nodes, tracker.input_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 1)
    
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    # Test forward pass
    print(f"\nTesting forward pass with {num_nodes} nodes...")
    outputs = tracker(graph)
    
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} tensors")
    
    print("✓ GNN tracker test completed")