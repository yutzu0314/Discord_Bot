"""
Centralized Settings for Smart Hungarian Speed Estimation
========================================================

Speed-focused configuration without MOT17 dataset dependency.
Optimized untuk video input dan speed estimation dengan VisDrone weights.

SIMPLE CONFIGURATION:
1. Set your model weights path  
2. Everything else is handled automatically

USAGE:
from utils.settings import SETTINGS
"""

import os
from pathlib import Path
import torch

class Settings:
    """
    Centralized settings untuk Smart Hungarian Speed Estimation
    Focused pada video processing dan speed calculation
    """
    
    # =====================================
    # USER CONFIGURATION - MODIFY THESE
    # =====================================

    # Model Weights Configuration
    DETECTOR_WEIGHTS = "best_visdrone_full.pt"  # YOLOv8 weights path
    REID_MODEL = "mobilenetv3_large_100"  # ReID model name

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Performance Preset
    PERFORMANCE_PRESET = "balanced"  # Options: "high_accuracy", "balanced", "high_recall"
    
    # Speed Estimation Configuration
    ENABLE_SPEED_CALCULATION = True
    DEFAULT_PIXELS_PER_METER = 20.0
    DEFAULT_SPEED_UNIT = "kmh"  # Options: "kmh", "ms"
    
    # =====================================
    # AUTOMATIC CONFIGURATION - DON'T MODIFY
    # =====================================
    
    def __init__(self):
        """Initialize and validate settings"""
        self._setup_output_dirs()
        self._setup_derived_configs()
    
    def _setup_output_dirs(self):
        """Setup output directories"""
        self.OUTPUT_ROOT = "results"
        os.makedirs(self.OUTPUT_ROOT, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_ROOT, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_ROOT, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_ROOT, "speed_analysis"), exist_ok=True)
        
        print(f"Output directory created: {self.OUTPUT_ROOT}")
    
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
            'half_precision': True,
            # VisDrone specific classes
            'target_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # All VisDrone classes
            'vehicle_classes': [4, 5, 6, 7, 8, 9],  # car, van, truck, tricycle, awning-tricycle, bus
            'person_classes': [0, 1],  # pedestrian, people
            'bike_classes': [2, 3]  # bicycle, motorcycle
        }
        
        # =====================================
        # REID CONFIGURATION  
        # =====================================
        self.REID_CONFIG = {
            'model_name': self.REID_MODEL,
            'feature_dim': 128,
            'device': self.DEVICE,
            'image_size': (256, 128),  # (Height, Width)
            'normalize': True,
            'extraction_layer': 'features'
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
        # SMART HUNGARIAN CONFIGURATION
        # =====================================
        self.SMART_HUNGARIAN_CONFIG = {
            # Quality thresholds berdasarkan scene complexity
            'uncertainty_thresholds': {
                'sparse': 0.25,     # More permissive for sparse scenes
                'normal': 0.18,     # Balanced for normal scenes
                'crowded': 0.08,    # Stricter for crowded scenes
                'extreme': 0.06     # Very strict for extreme scenes
            },
            
            # Spatial constraints
            'max_distance': 150.0,
            'max_motion_error': 80.0,
            'bbox_overlap_threshold': 0.1,
            
            # Quality requirements
            'min_detection_confidence': 0.3,
            'min_track_stability': 0.2,
            'motion_consistency_age': 3,
            
            # Scene analysis parameters
            'scene_complexity_factors': {
                'object_density_weight': 0.35,
                'imbalance_weight': 0.25,
                'quality_variance_weight': 0.20,
                'uncertainty_complexity_weight': 0.20
            }
        }
        
        # =====================================
        # SPEED ESTIMATION CONFIGURATION
        # =====================================
        self.SPEED_CONFIG = {
            'enabled': self.ENABLE_SPEED_CALCULATION,
            'pixels_per_meter': self.DEFAULT_PIXELS_PER_METER,
            'speed_unit': self.DEFAULT_SPEED_UNIT,
            'fps': 30.0,
            
            # Physics constraints per object type
            'max_acceleration': {
                'person': 5.0,      # m/s²
                'car': 3.0,         # m/s²
                'truck': 2.5,       # m/s²
                'bus': 2.0,         # m/s²
                'motorcycle': 4.0,  # m/s²
                'bicycle': 4.0,     # m/s²
                'default': 3.0      # m/s²
            },
            
            # Confidence weighting
            'min_confidence_threshold': 0.3,
            'confidence_weight_exp': 2.0,
            
            # Temporal smoothing
            'smoothing_window': 5,
            'temporal_weight_decay': 0.8,
            
            # Adaptive pixel ratio
            'use_adaptive_pixel_ratio': True,
            'min_pixel_ratio': 15.0,  # pixels/meter at bottom (close)
            'max_pixel_ratio': 35.0,  # pixels/meter at top (far)
            
            # Motion predictor integration
            'use_kalman_velocity': True,
            'kalman_priority': True,  # Kalman velocity gets highest priority
            'fallback_methods': ['temporal', 'simple']
        }
        
        # =====================================
        # MOTION PREDICTOR CONFIGURATION
        # =====================================
        self.MOTION_PREDICTOR_CONFIG = {
            'enabled': True,
            'fps': 30.0,
            'min_track_hits': 3,
            'max_prediction_frames': 5,
            
            # Kalman filter parameters
            'process_noise': {
                'position': 0.1,
                'size': 0.1,
                'velocity': 8.0,
                'size_velocity': 0.5
            },
            
            'measurement_noise': 2.0,
            'initial_covariance': 50.0,
            
            # State model
            'state_dim': 8,  # [center_x, center_y, width, height, vel_x, vel_y, vel_w, vel_h]
            'obs_dim': 4     # [center_x, center_y, width, height]
        }
        
        # =====================================
        # TRACKING CONFIGURATION
        # =====================================
        tracking_params = self._get_tracking_params_by_preset(self.PERFORMANCE_PRESET)

        self.TRACKING_CONFIG = {
            # Association method
            'association_method': 'smart_hungarian',
            'max_association_cost': tracking_params['max_association_cost'],
            
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
            
            # Quality gates
            'appearance_gate_threshold': tracking_params['appearance_gate_threshold'],
            'spatial_gate_threshold': tracking_params['spatial_gate_threshold'],
            'min_iou_threshold': tracking_params['min_iou_threshold']
        }
        
        # =====================================
        # VISUALIZATION CONFIGURATION
        # =====================================
        self.VISUALIZATION_CONFIG = {
            # Basic visualization
            'show_detections': True,
            'show_tracks': True,
            'show_track_ids': True,
            'show_ghost_nodes': True,
            
            # Speed-specific visualization
            'show_speed_info': True,
            'show_direction_arrows': True,
            'show_speed_trails': True,
            'show_virtual_lines': True,
            
            # Colors
            'detection_color': (0, 255, 0),
            'track_color': (255, 0, 0),
            'ghost_color': (0, 0, 255),
            'speed_color_coding': True,
            
            # UI elements
            'bbox_thickness': 2,
            'font_scale': 0.5,
            'arrow_length': 30,
            'trail_length': 20,
            
            # Smart Hungarian specific
            'show_smart_hungarian_info': True,
            'show_quality_scores': False,  # Set True for debugging
            'uncertainty_color_map': False
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
            # Balanced settings optimized for speed estimation
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
    
    # =====================================
    # CONFIGURATION GETTERS
    # =====================================
    
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
    
    def get_smart_hungarian_config(self):
        """Get configuration for Smart Hungarian optimizer"""
        return self.SMART_HUNGARIAN_CONFIG
    
    def get_speed_config(self):
        """Get configuration for speed estimation"""
        return self.SPEED_CONFIG
    
    def get_motion_predictor_config(self):
        """Get configuration for motion predictor"""
        return self.MOTION_PREDICTOR_CONFIG
    
    def get_tracking_config(self):
        """Get configuration for track manager"""
        return self.TRACKING_CONFIG
    
    def get_visualization_config(self):
        """Get configuration for visualization"""
        return self.VISUALIZATION_CONFIG
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("SMART HUNGARIAN SPEED ESTIMATION CONFIGURATION")
        print("="*60)
        
        print(f"Models:")
        print(f"   Detector: {self.DETECTOR_WEIGHTS}")
        print(f"   ReID: {self.REID_MODEL}")
        print(f"   Device: {self.DEVICE}")
        
        print(f"\nSpeed Estimation:")
        print(f"   Enabled: {'Yes' if self.SPEED_CONFIG['enabled'] else 'No'}")
        print(f"   Default calibration: {self.SPEED_CONFIG['pixels_per_meter']} px/m")
        print(f"   Default unit: {self.SPEED_CONFIG['speed_unit'].upper()}")
        print(f"   Kalman integration: {'Yes' if self.SPEED_CONFIG['use_kalman_velocity'] else 'No'}")
        
        print(f"\nSmart Hungarian:")
        print(f"   Association method: {self.TRACKING_CONFIG['association_method']}")
        print(f"   Scene-adaptive thresholds: Yes")
        print(f"   Quality control: 5-layer gate system")
        
        print(f"\nTracking:")
        print(f"   Performance preset: {self.PERFORMANCE_PRESET}")
        print(f"   Max association cost: {self.TRACKING_CONFIG['max_association_cost']}")
        print(f"   ReID threshold: {self.TRACKING_CONFIG['reid_threshold']}")
        print(f"   Ghost nodes: {'Yes' if self.TRACKING_CONFIG['enable_ghost_nodes'] else 'No'}")
        
        print(f"\nOutput:")
        print(f"   Results directory: {self.OUTPUT_ROOT}")
        
        print("="*60)
    
    def create_config_object(self):
        """Create a config object compatible dengan existing code"""
        class ConfigObject:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, ConfigObject(value))
                    else:
                        setattr(self, key, value)
        
        # Create config structure untuk backward compatibility
        full_config = {
            'detector': self.DETECTOR_CONFIG,
            'reid': self.REID_CONFIG,
            'graph': self.GRAPH_CONFIG,
            'gnn': self.GNN_CONFIG,
            'smart_hungarian': self.SMART_HUNGARIAN_CONFIG,
            'speed': self.SPEED_CONFIG,
            'motion_predictor': self.MOTION_PREDICTOR_CONFIG,
            'tracking': self.TRACKING_CONFIG,
            'visualization': self.VISUALIZATION_CONFIG
        }
        
        return ConfigObject(full_config)


# =====================================
# GLOBAL SETTINGS INSTANCE
# =====================================

# Create global settings instance
SETTINGS = Settings()

# Backward compatibility functions
def get_config():
    """Get config object compatible dengan existing code"""
    return SETTINGS.create_config_object()

def get_device():
    """Get device"""
    return SETTINGS.DEVICE

def get_detector_weights():
    """Get detector weights path"""
    return SETTINGS.DETECTOR_WEIGHTS

def get_reid_model():
    """Get ReID model name"""
    return SETTINGS.REID_MODEL


# =====================================
# CONFIGURATION MODIFICATION HELPERS
# =====================================

def update_detector_weights(new_weights):
    """Update detector weights"""
    global SETTINGS
    SETTINGS.DETECTOR_WEIGHTS = new_weights
    SETTINGS.DETECTOR_CONFIG['model_variant'] = new_weights
    print(f"Detector weights updated to: {new_weights}")

def update_reid_model(new_model):
    """Update ReID model"""
    global SETTINGS
    SETTINGS.REID_MODEL = new_model
    SETTINGS.REID_CONFIG['model_name'] = new_model
    print(f"ReID model updated to: {new_model}")

def update_device(new_device):
    """Update device"""
    global SETTINGS
    SETTINGS.DEVICE = new_device
    SETTINGS.DETECTOR_CONFIG['device'] = new_device
    SETTINGS.REID_CONFIG['device'] = new_device
    SETTINGS.GRAPH_CONFIG['device'] = new_device
    SETTINGS.GNN_CONFIG['device'] = new_device
    print(f"Device updated to: {new_device}")

def update_speed_config(pixels_per_meter=None, speed_unit=None, enable_speed=None):
    """Update speed estimation configuration"""
    global SETTINGS
    
    if pixels_per_meter is not None:
        SETTINGS.SPEED_CONFIG['pixels_per_meter'] = pixels_per_meter
        SETTINGS.DEFAULT_PIXELS_PER_METER = pixels_per_meter
    
    if speed_unit is not None:
        SETTINGS.SPEED_CONFIG['speed_unit'] = speed_unit
        SETTINGS.DEFAULT_SPEED_UNIT = speed_unit
    
    if enable_speed is not None:
        SETTINGS.SPEED_CONFIG['enabled'] = enable_speed
        SETTINGS.ENABLE_SPEED_CALCULATION = enable_speed
    
    print(f"Speed config updated: {SETTINGS.SPEED_CONFIG['pixels_per_meter']} px/m, {SETTINGS.SPEED_CONFIG['speed_unit']}, enabled={SETTINGS.SPEED_CONFIG['enabled']}")

def update_smart_hungarian_thresholds(sparse=None, normal=None, crowded=None, extreme=None):
    """Update Smart Hungarian uncertainty thresholds"""
    global SETTINGS
    
    thresholds = SETTINGS.SMART_HUNGARIAN_CONFIG['uncertainty_thresholds']
    
    if sparse is not None:
        thresholds['sparse'] = sparse
    if normal is not None:
        thresholds['normal'] = normal
    if crowded is not None:
        thresholds['crowded'] = crowded
    if extreme is not None:
        thresholds['extreme'] = extreme
    
    print(f"Smart Hungarian thresholds updated: sparse={thresholds['sparse']}, normal={thresholds['normal']}, crowded={thresholds['crowded']}, extreme={thresholds['extreme']}")

def enable_debug_mode():
    """Enable debug visualization"""
    global SETTINGS
    SETTINGS.VISUALIZATION_CONFIG['show_quality_scores'] = True
    SETTINGS.VISUALIZATION_CONFIG['uncertainty_color_map'] = True
    print("Debug mode enabled - quality scores and uncertainty visualization active")

def disable_debug_mode():
    """Disable debug visualization"""
    global SETTINGS
    SETTINGS.VISUALIZATION_CONFIG['show_quality_scores'] = False
    SETTINGS.VISUALIZATION_CONFIG['uncertainty_color_map'] = False
    print("Debug mode disabled")


# =====================================
# PRESET CONFIGURATIONS
# =====================================

def apply_high_accuracy_preset():
    """Apply high accuracy preset untuk minimal ID switches"""
    global SETTINGS
    SETTINGS.PERFORMANCE_PRESET = "high_accuracy"
    SETTINGS._setup_derived_configs()
    
    # Stricter Smart Hungarian thresholds
    update_smart_hungarian_thresholds(
        sparse=0.20, normal=0.12, crowded=0.06, extreme=0.04
    )
    
    print("High accuracy preset applied - stricter thresholds for minimal ID switches")

def apply_high_recall_preset():
    """Apply high recall preset untuk maksimal detection"""
    global SETTINGS
    SETTINGS.PERFORMANCE_PRESET = "high_recall"
    SETTINGS._setup_derived_configs()
    
    # Relaxed Smart Hungarian thresholds
    update_smart_hungarian_thresholds(
        sparse=0.35, normal=0.25, crowded=0.12, extreme=0.08
    )
    
    print("High recall preset applied - relaxed thresholds for maximum detection")

def apply_balanced_preset():
    """Apply balanced preset (default)"""
    global SETTINGS
    SETTINGS.PERFORMANCE_PRESET = "balanced"
    SETTINGS._setup_derived_configs()
    
    # Balanced Smart Hungarian thresholds
    update_smart_hungarian_thresholds(
        sparse=0.25, normal=0.18, crowded=0.08, extreme=0.06
    )
    
    print("Balanced preset applied - optimized for speed estimation")

def apply_speed_optimized_preset():
    """Apply preset optimized specifically for speed estimation"""
    global SETTINGS
    
    # Use balanced tracking with speed-specific optimizations
    apply_balanced_preset()
    
    # Optimize for speed estimation
    SETTINGS.SPEED_CONFIG.update({
        'confidence_weight_exp': 1.8,  # More aggressive confidence weighting
        'smoothing_window': 7,         # Longer smoothing for better accuracy
        'use_adaptive_pixel_ratio': True,
        'kalman_priority': True
    })
    
    # Optimize motion predictor for speed
    SETTINGS.MOTION_PREDICTOR_CONFIG.update({
        'min_track_hits': 2,  # Earlier velocity calculation
        'max_prediction_frames': 7
    })
    
    print("Speed-optimized preset applied - enhanced speed estimation accuracy")


# Example usage and testing
if __name__ == "__main__":
    # Test speed-focused settings
    print("Testing Smart Hungarian Speed Estimation Settings...")
    
    # Print configuration summary
    SETTINGS.print_summary()
    
    # Test speed configuration
    print(f"\nTesting Speed Configuration:")
    speed_config = SETTINGS.get_speed_config()
    print(f"Enabled: {speed_config['enabled']}")
    print(f"Default calibration: {speed_config['pixels_per_meter']} px/m")
    print(f"Default unit: {speed_config['speed_unit']}")
    print(f"Kalman integration: {speed_config['use_kalman_velocity']}")
    
    # Test Smart Hungarian configuration
    print(f"\nTesting Smart Hungarian Configuration:")
    sh_config = SETTINGS.get_smart_hungarian_config()
    print(f"Uncertainty thresholds: {sh_config['uncertainty_thresholds']}")
    print(f"Max distance: {sh_config['max_distance']}")
    print(f"Quality requirements: {sh_config['min_detection_confidence']}")
    
    # Test preset application
    print(f"\nTesting Presets:")
    apply_speed_optimized_preset()
    print(f"Speed optimized - Smoothing window: {SETTINGS.SPEED_CONFIG['smoothing_window']}")
    
    apply_balanced_preset()
    print(f"Balanced mode - Normal threshold: {SETTINGS.SMART_HUNGARIAN_CONFIG['uncertainty_thresholds']['normal']}")
    
    # Test config object creation
    config = get_config()
    print(f"\nConfig object created: {type(config)}")
    print(f"Detector confidence: {config.detector.confidence_threshold}")
    print(f"Speed enabled: {config.speed.enabled}")
    print(f"Smart Hungarian max distance: {config.smart_hungarian.max_distance}")
    
    # Test configuration updates
    print(f"\nTesting Configuration Updates:")
    update_speed_config(pixels_per_meter=25.0, speed_unit="ms")
    print(f"Updated speed config: {SETTINGS.SPEED_CONFIG['pixels_per_meter']} px/m, {SETTINGS.SPEED_CONFIG['speed_unit']}")
    
    update_smart_hungarian_thresholds(normal=0.15, crowded=0.07)
    print(f"Updated thresholds: {SETTINGS.SMART_HUNGARIAN_CONFIG['uncertainty_thresholds']}")
    
    print("\nSmart Hungarian Speed Estimation settings test completed!")
    print(f"Ready for video input processing dengan VisDrone weights!")