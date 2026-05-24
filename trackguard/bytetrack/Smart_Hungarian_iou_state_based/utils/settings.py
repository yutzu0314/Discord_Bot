"""
Centralized Settings for GBC-MOT POC with AGM Support
Single source of truth for all configuration parameters including AGM

SIMPLE CONFIGURATION:
1. Set your dataset path
2. Set your model weights path  
3. Everything else is handled automatically

USAGE:
from utils.settings import SETTINGS
"""

import os
from pathlib import Path
import torch

class Settings:
    """
    Centralized settings for GBC-MOT POC with AGM integration
    Modify these values according to your setup
    """
    
   # =====================================
    # USER CONFIGURATION - MODIFY THESE
    # =====================================

    # Dataset Configuration
    DATASET_ROOT = r"E:\Users\phantom\Backup Riset\TrackGraph-SHA\MOT17-SDP\train"
    SEQUENCE_NAME = "MOT17-09-SDP"  # Will auto-detect if not found

    # Model Weights Configuration
    DETECTOR_WEIGHTS = "best.pt"  # YOLOv8 weights path (put your custom weights here)
    REID_MODEL = "mobilenetv3_large_100"  # ReID model name

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Performance Preset (change this to adjust all parameters at once)
    PERFORMANCE_PRESET = "balanced"  # Options: "high_accuracy", "balanced", "high_recall"
    
    # AGM Configuration
    AGM_MODE = True  # Set to True to use AGM, False for traditional matching
    
    # =====================================
    # AUTOMATIC CONFIGURATION - DON'T MODIFY
    # =====================================
    
    def __init__(self):
        """Initialize and validate settings"""
        self._validate_paths()
        self._auto_detect_sequence()
        self._setup_derived_configs()
    
    def _validate_paths(self):
        """Validate and create necessary paths"""
        # Validate dataset root
        if not os.path.exists(self.DATASET_ROOT):
            raise FileNotFoundError(
                f"Dataset root not found: {self.DATASET_ROOT}\n"
                f"Please update DATASET_ROOT in utils/settings.py"
            )
        
        # Create output directories
        self.OUTPUT_ROOT = "results"
        os.makedirs(self.OUTPUT_ROOT, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_ROOT, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_ROOT, "logs"), exist_ok=True)
        
        print(f"✓ Dataset root validated: {self.DATASET_ROOT}")
        print(f"✓ Output directory created: {self.OUTPUT_ROOT}")
    
    def _auto_detect_sequence(self):
        """Auto-detect available sequences and validate selected sequence"""
        available_sequences = self._get_available_sequences()
        
        if not available_sequences:
            raise FileNotFoundError(
                f"No MOT17 sequences found in {self.DATASET_ROOT}\n"
                f"Expected structure: dataset_root/train/MOT17-XX-YYY/ or dataset_root/MOT17-XX-YYY/"
            )
        
        print(f"Available sequences: {available_sequences}")
        
        # Check if selected sequence exists
        sequence_found = False
        possible_paths = [
            os.path.join(self.DATASET_ROOT, "train", self.SEQUENCE_NAME),
            os.path.join(self.DATASET_ROOT, self.SEQUENCE_NAME)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.SEQUENCE_PATH = path
                sequence_found = True
                break
        
        if not sequence_found:
            # Auto-select first available sequence
            old_name = self.SEQUENCE_NAME
            self.SEQUENCE_NAME = available_sequences[0]
            
            # Set correct path for auto-selected sequence
            possible_paths = [
                os.path.join(self.DATASET_ROOT, "train", self.SEQUENCE_NAME),
                os.path.join(self.DATASET_ROOT, self.SEQUENCE_NAME)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.SEQUENCE_PATH = path
                    break
            
            print(f"⚠️  {old_name} not found. Auto-selected: {self.SEQUENCE_NAME}")
        
        print(f"✓ Using sequence: {self.SEQUENCE_NAME}")
        print(f"✓ Sequence path: {self.SEQUENCE_PATH}")
        
        # Set derived paths
        self.IMAGES_PATH = os.path.join(self.SEQUENCE_PATH, "img1")
        self.GT_PATH = os.path.join(self.SEQUENCE_PATH, "gt", "gt.txt")
        
        if not os.path.exists(self.IMAGES_PATH):
            raise FileNotFoundError(f"Images folder not found: {self.IMAGES_PATH}")
        
        if not os.path.exists(self.GT_PATH):
            print(f"⚠️  Ground truth not found: {self.GT_PATH}")
    
    def _get_available_sequences(self):
        """Get list of available MOT17 sequences"""
        sequences = []
        
        # Try train folder structure first
        train_path = os.path.join(self.DATASET_ROOT, "train")
        if os.path.exists(train_path):
            for item in os.listdir(train_path):
                item_path = os.path.join(train_path, item)
                if os.path.isdir(item_path) and item.startswith("MOT17"):
                    sequences.append(item)
        
        # If no train folder, check direct structure
        if not sequences and os.path.exists(self.DATASET_ROOT):
            for item in os.listdir(self.DATASET_ROOT):
                item_path = os.path.join(self.DATASET_ROOT, item)
                if os.path.isdir(item_path) and item.startswith("MOT17"):
                    sequences.append(item)
        
        return sorted(sequences)
    
    def _setup_derived_configs(self):
        """Setup derived configuration parameters"""
        
        # =====================================
        # DETECTOR CONFIGURATION
        # =====================================
        self.DETECTOR_CONFIG = {
            'model_variant': self.DETECTOR_WEIGHTS,
            'confidence_threshold': 0.25,
            'nms_threshold': 0.3,
            'device': self.DEVICE,
            'input_size': 640,
            'half_precision': True
        }
        
        # =====================================
        # REID CONFIGURATION  
        # =====================================
        self.REID_CONFIG = {
            'model_name': self.REID_MODEL,
            'feature_dim': 128,
            'device': self.DEVICE,
            'image_size': (256, 128)  # (Height, Width)
        }
        
        # =====================================
        # GRAPH CONFIGURATION
        # =====================================
        self.GRAPH_CONFIG = {
            # Node representation dimensions
            'position_dim': 2,
            'size_dim': 2,
            'appearance_dim': 128,
            'hidden_dim': 64,
            'temporal_dim': 1,
            
            # Edge computation weights
            'distance_weight': 0.8,
            'similarity_weight': 1.5,
            'temporal_weight': 1.0,
            'motion_weight': 0.7,
            
            # Graph construction thresholds
            'max_distance_threshold': 120.0,
            'min_similarity_threshold': 0.35,
            'max_neighbors': 8,
            'use_knn_graph': True,
            'k_neighbors': 4,
            'epsilon': 1e-8,
            'device': self.DEVICE
        }
        
        # =====================================
        # GNN CONFIGURATION
        # =====================================
        self.GNN_CONFIG = {
            'model_type': 'GAT',
            'num_layers': 2,
            'hidden_dim': 64,
            'num_heads': 4,
            'dropout': 0.15,
            'aggregation': 'mean',
            'activation': 'relu',
            'predict_position': True,
            'predict_size': True,
            'prediction_dim': 4,
            'device': self.DEVICE,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        }
        
        # =====================================
        # AGM CONFIGURATION - NEW!
        # =====================================
        self.AGM_CONFIG = {
            # Core AGM settings
            'enable_detailed_logging': False,  # Set True for debugging
            'validate_assignments': True,      # Validate assignment correctness
            
            # Multi-tier uncertainty thresholds
            'tier1_threshold': 0.2,    # Ultra-confident assignments
            'tier2_threshold': 0.4,    # High-confident assignments  
            'tier3_threshold': 0.6,    # Medium-confident assignments
            'tier4_threshold': 0.8,    # Gap-filling assignments
            
            # Uncertainty calculation weights for new tracks
            'new_track_weights': {
                'iou': 0.6,
                'distance': 0.3,
                'confidence': 0.1,
                'appearance': 0.0,
                'motion': 0.0,
                'temporal': 0.0
            },
            
            # Uncertainty calculation weights for established tracks
            'established_track_weights': {
                'iou': 0.4,
                'distance': 0.25,
                'motion': 0.2,
                'appearance': 0.1,
                'confidence': 0.05,
                'temporal': 0.0
            },
            
            # Normalization parameters
            'max_distance': 500.0,
            'max_velocity_diff': 50.0,
            'iou_boost_factor': 1.5,
            
            # Scene complexity factors
            'complexity_factors': {
                'imbalance_weight': 0.3,
                'density_weight': 0.4,
                'confidence_weight': 0.3
            },
            
            # Performance monitoring
            'track_performance_history': True,
            'max_history_length': 100
        }
        
        # =====================================
        # TRACKING CONFIGURATION
        # =====================================
        # Apply preset-based configuration
        tracking_params = self._get_tracking_params_by_preset(self.PERFORMANCE_PRESET)

        self.TRACKING_CONFIG = {
            # Association method selection
            'association_method': 'agm' if self.AGM_MODE else 'hungarian',
            'max_association_cost': tracking_params['max_association_cost'],

            # ByteTrack-style association (NEW!)
            'use_bytetrack_association': False,  # Set True to enable ByteTrack three-stage matching

            # Track lifecycle
            'min_track_length': tracking_params['min_track_length'],
            'max_missing_frames': tracking_params['max_missing_frames'],
            'track_buffer_size': tracking_params['track_buffer_size'],

            # Ghost node management
            'enable_ghost_nodes': True,
            'ghost_prediction_method': 'kalman',
            'ghost_decay_factor': tracking_params['ghost_decay_factor'],

            # Re-identification
            'reid_threshold': tracking_params['reid_threshold'],
            'reid_time_window': tracking_params['reid_time_window'],

            # Gating thresholds (used by AGM as well)
            'appearance_gate_threshold': tracking_params['appearance_gate_threshold'],
            'spatial_gate_threshold': tracking_params['spatial_gate_threshold'],
            'min_iou_threshold': tracking_params['min_iou_threshold']
        }
        
        # =====================================
        # VISUALIZATION CONFIGURATION
        # =====================================
        self.VISUALIZATION_CONFIG = {
            'show_detections': True,
            'show_tracks': True,
            'show_track_ids': True,
            'show_graph_edges': True,
            'show_ghost_nodes': True,
            'detection_color': (0, 255, 0),
            'track_color': (255, 0, 0),
            'ghost_color': (0, 0, 255),
            'edge_color': (255, 255, 0),
            'bbox_thickness': 2,
            'font_scale': 0.5,
            # AGM-specific visualization
            'show_agm_info': self.AGM_MODE,
            'agm_color': (255, 255, 0),
            'uncertainty_color_map': True
        }
        
        # =====================================
        # EXPERIMENT CONFIGURATION
        # =====================================
        self.EXPERIMENT_CONFIG = {
            'start_frame': 1,
            'end_frame': 600,
            'frame_step': 1,
            'save_results': True,
            'save_visualizations': True,
            'compute_metrics': True,
            'log_level': 'INFO',
            # AGM-specific experiment settings
            'agm_performance_tracking': self.AGM_MODE,
            'agm_detailed_analysis': False,  # Set True for research analysis
            'save_agm_statistics': True
        }

    def _get_tracking_params_by_preset(self, preset: str) -> dict:
        """Get tracking parameters based on performance preset"""
        if preset == "high_accuracy":
            # Strictest settings for minimum ID switches
            return {
                'max_association_cost': 0.4,
                'min_track_length': 3,
                'max_missing_frames': 20,
                'track_buffer_size': 6,
                'ghost_decay_factor': 0.8,
                'reid_threshold': 0.75,
                'reid_time_window': 20,
                'appearance_gate_threshold': 0.8,
                'spatial_gate_threshold': 200.0,
                'min_iou_threshold': 0.1
            }
        elif preset == "high_recall":
            # Relaxed settings for maximum detection
            return {
                'max_association_cost': 0.7,
                'min_track_length': 2,
                'max_missing_frames': 35,
                'track_buffer_size': 12,
                'ghost_decay_factor': 0.9,
                'reid_threshold': 0.55,
                'reid_time_window': 35,
                'appearance_gate_threshold': 0.65,
                'spatial_gate_threshold': 300.0,
                'min_iou_threshold': 0.01
            }
        else:  # "balanced"
            # Balanced settings (optimized from config_tuned.py)
            return {
                'max_association_cost': 0.5,
                'min_track_length': 3,
                'max_missing_frames': 25,
                'track_buffer_size': 8,
                'ghost_decay_factor': 0.85,
                'reid_threshold': 0.65,
                'reid_time_window': 25,
                'appearance_gate_threshold': 0.75,
                'spatial_gate_threshold': 250.0,
                'min_iou_threshold': 0.05
            }
    
    def get_detector_config(self):
        """Get configuration for detector"""
        return self.DETECTOR_CONFIG
    
    def get_reid_config(self):
        """Get configuration for ReID extractor"""
        return self.REID_CONFIG
    
    def get_graph_config(self):
        """Get configuration for graph builder"""
        return self.GRAPH_CONFIG
    
    def get_gnn_config(self):
        """Get configuration for GNN tracker"""
        return self.GNN_CONFIG
    
    def get_agm_config(self):
        """Get configuration for AGM matcher - NEW!"""
        return self.AGM_CONFIG
    
    def get_tracking_config(self):
        """Get configuration for track manager"""
        return self.TRACKING_CONFIG
    
    def get_visualization_config(self):
        """Get configuration for visualization"""
        return self.VISUALIZATION_CONFIG
    
    def get_experiment_config(self):
        """Get configuration for experiments"""
        return self.EXPERIMENT_CONFIG
    
    def get_dataset_info(self):
        """Get dataset information"""
        return {
            'dataset_root': self.DATASET_ROOT,
            'sequence_name': self.SEQUENCE_NAME,
            'sequence_path': self.SEQUENCE_PATH,
            'images_path': self.IMAGES_PATH,
            'gt_path': self.GT_PATH,
            'has_gt': os.path.exists(self.GT_PATH)
        }
    
    def print_summary(self):
        """Print configuration summary with AGM info"""
        print("\n" + "="*60)
        print("🔧 GBC-MOT POC CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"📁 Dataset:")
        print(f"   Root: {self.DATASET_ROOT}")
        print(f"   Sequence: {self.SEQUENCE_NAME}")
        print(f"   Images: {self.IMAGES_PATH}")
        print(f"   GT: {'✓' if os.path.exists(self.GT_PATH) else '✗'}")
        
        print(f"\n🤖 Models:")
        print(f"   Detector: {self.DETECTOR_WEIGHTS}")
        print(f"   ReID: {self.REID_MODEL}")
        print(f"   Device: {self.DEVICE}")
        
        print(f"\n🎯 Matching Algorithm:")
        print(f"   AGM Mode: {'✓' if self.AGM_MODE else '✗'}")
        print(f"   Association Method: {self.TRACKING_CONFIG['association_method']}")
        if self.AGM_MODE:
            print(f"   AGM Tier1 Threshold: {self.AGM_CONFIG['tier1_threshold']}")
            print(f"   AGM Tier4 Threshold: {self.AGM_CONFIG['tier4_threshold']}")
            print(f"   AGM Detailed Logging: {'✓' if self.AGM_CONFIG['enable_detailed_logging'] else '✗'}")
        
        print(f"\n⚙️  Key Parameters:")
        print(f"   Association Cost: {self.TRACKING_CONFIG['max_association_cost']}")
        print(f"   ReID Threshold: {self.TRACKING_CONFIG['reid_threshold']}")
        print(f"   Ghost Nodes: {'✓' if self.TRACKING_CONFIG['enable_ghost_nodes'] else '✗'}")
        
        print(f"\n📊 Experiment:")
        print(f"   Frames: {self.EXPERIMENT_CONFIG['start_frame']}-{self.EXPERIMENT_CONFIG['end_frame']}")
        print(f"   Output: {self.OUTPUT_ROOT}")
        print(f"   AGM Performance Tracking: {'✓' if self.EXPERIMENT_CONFIG['agm_performance_tracking'] else '✗'}")
        
        print("="*60)
    
    def create_config_object(self):
        """Create a config object compatible with existing code"""
        class ConfigObject:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, ConfigObject(value))
                    else:
                        setattr(self, key, value)
        
        # Create nested config structure
        full_config = {
            'dataset': {
                'dataset_root': self.DATASET_ROOT,
                'sequence_name': self.SEQUENCE_NAME,
                'load_gt': True,
                'start_frame': self.EXPERIMENT_CONFIG['start_frame'],
                'end_frame': self.EXPERIMENT_CONFIG['end_frame']
            },
            'detector': self.DETECTOR_CONFIG,
            'reid': self.REID_CONFIG,
            'graph': self.GRAPH_CONFIG,
            'gnn': self.GNN_CONFIG,
            'agm': self.AGM_CONFIG,  # NEW: AGM configuration
            'tracking': self.TRACKING_CONFIG,
            'visualization': self.VISUALIZATION_CONFIG,
            'experiment': self.EXPERIMENT_CONFIG
        }
        
        return ConfigObject(full_config)


# =====================================
# GLOBAL SETTINGS INSTANCE
# =====================================

# Create global settings instance
SETTINGS = Settings()

# Backward compatibility functions
def get_config():
    """Get config object compatible with existing code"""
    return SETTINGS.create_config_object()

def get_tuned_config():
    """Get tuned config (same as get_config for centralized settings)"""
    return SETTINGS.create_config_object()

def get_dataset_path():
    """Get dataset path"""
    return SETTINGS.DATASET_ROOT

def get_sequence_name():
    """Get sequence name"""
    return SETTINGS.SEQUENCE_NAME

def get_device():
    """Get device"""
    return SETTINGS.DEVICE

def get_detector_weights():
    """Get detector weights path"""
    return SETTINGS.DETECTOR_WEIGHTS

def get_reid_model():
    """Get ReID model name"""
    return SETTINGS.REID_MODEL

def get_agm_config():
    """Get AGM configuration - NEW!"""
    return SETTINGS.get_agm_config()

def is_agm_enabled():
    """Check if AGM mode is enabled - NEW!"""
    return SETTINGS.AGM_MODE


# =====================================
# CONFIGURATION MODIFICATION HELPERS
# =====================================

def update_dataset_path(new_path):
    """Update dataset path"""
    global SETTINGS
    SETTINGS.DATASET_ROOT = new_path
    SETTINGS._validate_paths()
    SETTINGS._auto_detect_sequence()
    print(f"✓ Dataset path updated to: {new_path}")

def update_sequence(new_sequence):
    """Update sequence name"""
    global SETTINGS
    SETTINGS.SEQUENCE_NAME = new_sequence
    SETTINGS._auto_detect_sequence()
    print(f"✓ Sequence updated to: {new_sequence}")

def update_detector_weights(new_weights):
    """Update detector weights"""
    global SETTINGS
    SETTINGS.DETECTOR_WEIGHTS = new_weights
    SETTINGS.DETECTOR_CONFIG['model_variant'] = new_weights
    print(f"✓ Detector weights updated to: {new_weights}")

def update_reid_model(new_model):
    """Update ReID model"""
    global SETTINGS
    SETTINGS.REID_MODEL = new_model
    SETTINGS.REID_CONFIG['model_name'] = new_model
    print(f"✓ ReID model updated to: {new_model}")

def update_device(new_device):
    """Update device"""
    global SETTINGS
    SETTINGS.DEVICE = new_device
    SETTINGS.DETECTOR_CONFIG['device'] = new_device
    SETTINGS.REID_CONFIG['device'] = new_device
    SETTINGS.GRAPH_CONFIG['device'] = new_device
    SETTINGS.GNN_CONFIG['device'] = new_device
    print(f"✓ Device updated to: {new_device}")

def enable_agm_mode():
    """Enable AGM matching mode - NEW!"""
    global SETTINGS
    SETTINGS.AGM_MODE = True
    SETTINGS.TRACKING_CONFIG['association_method'] = 'agm'
    SETTINGS.EXPERIMENT_CONFIG['agm_performance_tracking'] = True
    SETTINGS.VISUALIZATION_CONFIG['show_agm_info'] = True
    print("✓ AGM mode enabled")

def disable_agm_mode():
    """Disable AGM matching mode - NEW!"""
    global SETTINGS
    SETTINGS.AGM_MODE = False
    SETTINGS.TRACKING_CONFIG['association_method'] = 'hungarian'
    SETTINGS.EXPERIMENT_CONFIG['agm_performance_tracking'] = False
    SETTINGS.VISUALIZATION_CONFIG['show_agm_info'] = False
    print("✓ AGM mode disabled - using traditional Hungarian")

def update_agm_thresholds(tier1=None, tier2=None, tier3=None, tier4=None):
    """Update AGM uncertainty thresholds - NEW!"""
    global SETTINGS
    
    if tier1 is not None:
        SETTINGS.AGM_CONFIG['tier1_threshold'] = tier1
    if tier2 is not None:
        SETTINGS.AGM_CONFIG['tier2_threshold'] = tier2
    if tier3 is not None:
        SETTINGS.AGM_CONFIG['tier3_threshold'] = tier3
    if tier4 is not None:
        SETTINGS.AGM_CONFIG['tier4_threshold'] = tier4
    
    print(f"✓ AGM thresholds updated: T1={SETTINGS.AGM_CONFIG['tier1_threshold']}, "
          f"T2={SETTINGS.AGM_CONFIG['tier2_threshold']}, T3={SETTINGS.AGM_CONFIG['tier3_threshold']}, "
          f"T4={SETTINGS.AGM_CONFIG['tier4_threshold']}")

def enable_agm_debug_mode():
    """Enable AGM detailed logging for debugging - NEW!"""
    global SETTINGS
    SETTINGS.AGM_CONFIG['enable_detailed_logging'] = True
    SETTINGS.EXPERIMENT_CONFIG['agm_detailed_analysis'] = True
    print("✓ AGM debug mode enabled - detailed logging active")

def disable_agm_debug_mode():
    """Disable AGM detailed logging - NEW!"""
    global SETTINGS
    SETTINGS.AGM_CONFIG['enable_detailed_logging'] = False
    SETTINGS.EXPERIMENT_CONFIG['agm_detailed_analysis'] = False
    print("✓ AGM debug mode disabled")

def enable_bytetrack_association():
    """Enable ByteTrack three-stage association - NEW!"""
    global SETTINGS
    SETTINGS.TRACKING_CONFIG['use_bytetrack_association'] = True
    print("✓ ByteTrack three-stage association enabled")
    print("  Stage 1: High-confidence detections (conf >= 0.5)")
    print("  Stage 2: Low-confidence detections (0.1 <= conf < 0.5)")
    print("  Stage 3: Ghost track recovery")

def disable_bytetrack_association():
    """Disable ByteTrack association, use Smart Hungarian - NEW!"""
    global SETTINGS
    SETTINGS.TRACKING_CONFIG['use_bytetrack_association'] = False
    print("✓ ByteTrack association disabled - using Smart Hungarian")


# =====================================
# QUICK CONFIGURATION PRESETS
# =====================================

def apply_high_accuracy_preset():
    """Apply high accuracy configuration preset"""
    global SETTINGS
    
    # Stricter thresholds for better precision
    SETTINGS.TRACKING_CONFIG.update({
        'max_association_cost': 0.4,
        'reid_threshold': 0.7,
        'appearance_gate_threshold': 0.8
    })
    
    SETTINGS.GRAPH_CONFIG.update({
        'min_similarity_threshold': 0.4,
        'max_distance_threshold': 100.0
    })
    
    # AGM high accuracy settings
    if SETTINGS.AGM_MODE:
        SETTINGS.AGM_CONFIG.update({
            'tier1_threshold': 0.15,  # More strict
            'tier2_threshold': 0.3,
            'tier3_threshold': 0.5,
            'tier4_threshold': 0.7
        })
    
    print("✓ High accuracy preset applied (stricter thresholds)")
    if SETTINGS.AGM_MODE:
        print("  AGM thresholds adjusted for high accuracy")

def apply_high_recall_preset():
    """Apply high recall configuration preset"""
    global SETTINGS
    
    # Relaxed thresholds for better recall
    SETTINGS.TRACKING_CONFIG.update({
        'max_association_cost': 0.6,
        'reid_threshold': 0.6,
        'appearance_gate_threshold': 0.7
    })
    
    SETTINGS.GRAPH_CONFIG.update({
        'min_similarity_threshold': 0.3,
        'max_distance_threshold': 150.0
    })
    
    # AGM high recall settings
    if SETTINGS.AGM_MODE:
        SETTINGS.AGM_CONFIG.update({
            'tier1_threshold': 0.25,  # More relaxed
            'tier2_threshold': 0.5,
            'tier3_threshold': 0.7,
            'tier4_threshold': 0.85
        })
    
    print("✓ High recall preset applied (relaxed thresholds)")
    if SETTINGS.AGM_MODE:
        print("  AGM thresholds adjusted for high recall")

def apply_balanced_preset():
    """Apply balanced configuration preset (default)"""
    global SETTINGS
    
    # Balanced thresholds
    SETTINGS.TRACKING_CONFIG.update({
        'max_association_cost': 0.5,
        'reid_threshold': 0.65,
        'appearance_gate_threshold': 0.75
    })
    
    SETTINGS.GRAPH_CONFIG.update({
        'min_similarity_threshold': 0.35,
        'max_distance_threshold': 120.0
    })
    
    # AGM balanced settings (default)
    if SETTINGS.AGM_MODE:
        SETTINGS.AGM_CONFIG.update({
            'tier1_threshold': 0.2,
            'tier2_threshold': 0.4,
            'tier3_threshold': 0.6,
            'tier4_threshold': 0.8
        })
    
    print("✓ Balanced preset applied (default settings)")
    if SETTINGS.AGM_MODE:
        print("  AGM thresholds set to balanced defaults")

def apply_agm_research_preset():
    """Apply AGM research configuration for maximum performance analysis - NEW!"""
    global SETTINGS
    
    if not SETTINGS.AGM_MODE:
        enable_agm_mode()
    
    # Research-oriented settings
    SETTINGS.AGM_CONFIG.update({
        'enable_detailed_logging': True,
        'validate_assignments': True,
        'track_performance_history': True,
        'max_history_length': 200,  # More history for research
        
        # Optimized thresholds for research
        'tier1_threshold': 0.18,
        'tier2_threshold': 0.38,
        'tier3_threshold': 0.58,
        'tier4_threshold': 0.78,
        
        # Enhanced uncertainty weights
        'iou_boost_factor': 1.8,
        'max_distance': 450.0,
        'max_velocity_diff': 45.0
    })
    
    SETTINGS.EXPERIMENT_CONFIG.update({
        'agm_detailed_analysis': True,
        'save_agm_statistics': True
    })
    
    print("✓ AGM research preset applied")
    print("  Detailed logging and analysis enabled")
    print("  Optimized thresholds for research performance")


# Example usage and testing
if __name__ == "__main__":
    # Test centralized settings with AGM
    print("Testing centralized settings with AGM support...")
    
    # Print configuration summary
    SETTINGS.print_summary()
    
    # Test AGM functions
    print(f"\nTesting AGM configuration:")
    print(f"AGM Mode: {is_agm_enabled()}")
    print(f"Association Method: {SETTINGS.TRACKING_CONFIG['association_method']}")
    
    # Test AGM config access
    agm_config = get_agm_config()
    print(f"AGM Tier1 Threshold: {agm_config['tier1_threshold']}")
    print(f"AGM Detailed Logging: {agm_config['enable_detailed_logging']}")
    
    # Test preset application
    print(f"\nTesting AGM presets:")
    apply_agm_research_preset()
    print(f"Research mode - Detailed logging: {SETTINGS.AGM_CONFIG['enable_detailed_logging']}")
    
    apply_balanced_preset()
    print(f"Balanced mode - Tier1 threshold: {SETTINGS.AGM_CONFIG['tier1_threshold']}")
    
    # Test config object creation
    config = get_config()
    print(f"Config object created: {type(config)}")
    print(f"Detector confidence: {config.detector.confidence_threshold}")
    print(f"Tracking cost: {config.tracking.max_association_cost}")
    
    # Test AGM config in object
    if hasattr(config, 'agm'):
        print(f"AGM config available: True")
        print(f"AGM Tier1: {config.agm.tier1_threshold}")
        print(f"AGM Validation: {config.agm.validate_assignments}")
    else:
        print(f"AGM config available: False")
    
    # Test AGM mode switching
    print(f"\nTesting AGM mode switching:")
    print(f"Current AGM mode: {is_agm_enabled()}")
    
    if is_agm_enabled():
        disable_agm_mode()
        print(f"After disable: {is_agm_enabled()}")
        print(f"Association method: {SETTINGS.TRACKING_CONFIG['association_method']}")
        
        enable_agm_mode()
        print(f"After enable: {is_agm_enabled()}")
        print(f"Association method: {SETTINGS.TRACKING_CONFIG['association_method']}")
    
    # Test AGM threshold updates
    print(f"\nTesting AGM threshold updates:")
    print(f"Original T1: {SETTINGS.AGM_CONFIG['tier1_threshold']}")
    
    update_agm_thresholds(tier1=0.15, tier2=0.35)
    print(f"Updated T1: {SETTINGS.AGM_CONFIG['tier1_threshold']}")
    print(f"Updated T2: {SETTINGS.AGM_CONFIG['tier2_threshold']}")
    
    # Test debug mode
    print(f"\nTesting AGM debug mode:")
    print(f"Debug mode before: {SETTINGS.AGM_CONFIG['enable_detailed_logging']}")
    
    enable_agm_debug_mode()
    print(f"Debug mode after enable: {SETTINGS.AGM_CONFIG['enable_detailed_logging']}")
    
    disable_agm_debug_mode()
    print(f"Debug mode after disable: {SETTINGS.AGM_CONFIG['enable_detailed_logging']}")
    
    print("\n✓ Centralized settings with AGM support test completed!")
    print(f"🎯 AGM Mode: {'ENABLED' if is_agm_enabled() else 'DISABLED'}")
    print(f"🎯 Ready for AGM evaluation pipeline")