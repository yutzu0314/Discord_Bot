"""
ReID Feature Extractor for GBC-MOT POC
Uses centralized settings from utils.settings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import timm
import warnings
import time

class MobileNetV3Extractor:
    """
    MobileNetV3-based appearance feature extractor untuk pedestrian re-identification
    Menggantikan OSNet dengan model yang lebih cocok untuk IoT applications
    """
    
    def __init__(self, **kwargs):
        """
        Initialize MobileNetV3 feature extractor using centralized settings
        """
        from utils.settings import SETTINGS
        
        # Get config from centralized settings
        config = SETTINGS.get_reid_config()
        
        # Override with any kwargs provided
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
        
        # Apply configuration
        self.model_name = config['model_name']
        self.feature_dim = config['feature_dim']
        self.device = config['device']
        self.image_size = config['image_size']
        
        # Load backbone model
        self.backbone = self._load_backbone_model()
        self.feature_reducer = self._build_feature_reducer()
        
        # Set ke evaluation mode
        self.backbone.eval()
        self.feature_reducer.eval()
        
        # Image preprocessing untuk ReID
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Performance tracking
        self.extraction_times = []
        self.total_extractions = 0
        
        print(f"✓ MobileNetV3 feature extractor loaded on {self.device}")
        print(f"  Model: {self.model_name}")
        print(f"  Output dim: {self.feature_dim}")
        print(f"  Input size: {self.image_size}")
        print(f"  Backbone output: {self._get_backbone_output_dim()}")
        print(f"  🎯 Using centralized settings")
    
    def _load_backbone_model(self) -> nn.Module:
        """Load MobileNetV3 backbone model"""
        try:
            # Coba load model utama
            model = timm.create_model(
                self.model_name, 
                pretrained=True, 
                num_classes=0  # Remove classification head
            )
            model.to(self.device)
            print(f"✓ Loaded {self.model_name} successfully")
            return model
            
        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            
            # Fallback ke EfficientNet-B0
            try:
                print("Trying EfficientNet-B0 fallback...")
                model = timm.create_model(
                    'efficientnet_b0', 
                    pretrained=True, 
                    num_classes=0
                )
                model.to(self.device)
                self.model_name = 'efficientnet_b0'
                print("✓ Using EfficientNet-B0 as backbone")
                return model
                
            except Exception as e2:
                print(f"EfficientNet-B0 failed: {e2}")
                
                # Final fallback ke ResNet18 (tapi dengan warning)
                print("⚠️ Using ResNet18 final fallback (not optimal for IoT)")
                import torchvision.models as models
                resnet = models.resnet18(weights='DEFAULT')
                backbone = nn.Sequential(*list(resnet.children())[:-1])
                backbone.to(self.device)
                self.model_name = 'resnet18'
                return backbone
    
    def _get_backbone_output_dim(self) -> int:
        """Get backbone output dimension"""
        # Test dengan dummy input
        dummy_input = torch.randn(1, 3, *self.image_size).to(self.device)
        with torch.no_grad():
            output = self.backbone(dummy_input)
            if len(output.shape) > 2:
                output = output.view(output.size(0), -1)
            return output.shape[1]
    
    def _build_feature_reducer(self) -> nn.Module:
        """Build feature dimension reducer dari backbone output ke target dim"""
        backbone_dim = self._get_backbone_output_dim()
        
        # Adaptive reducer berdasarkan backbone size
        if backbone_dim <= 512:
            # Small backbone (ResNet18)
            reducer = nn.Sequential(
                nn.Linear(backbone_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, self.feature_dim),
                L2Norm(dim=1)
            )
        else:
            # Large backbone (MobileNetV3, EfficientNet)
            reducer = nn.Sequential(
                nn.Linear(backbone_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, self.feature_dim),
                L2Norm(dim=1)
            )
        
        reducer.to(self.device)
        print(f"✓ Feature reducer: {backbone_dim} → {self.feature_dim}")
        return reducer
    
    def extract_features(self, image_crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract appearance features dari pedestrian crops
        
        Args:
            image_crops: List of pedestrian bounding box crops (BGR format)
            
        Returns:
            Feature matrix of shape [N, feature_dim] where N = len(image_crops)
        """
        if not image_crops:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        
        start_time = time.time()
        
        # Preprocess crops
        batch_tensors = []
        valid_indices = []
        
        for i, crop in enumerate(image_crops):
            try:
                # Convert BGR to RGB
                if len(crop.shape) == 3:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                else:
                    print(f"⚠️ Invalid crop shape: {crop.shape}, skipping")
                    continue
                
                # Apply transforms
                tensor = self.transform(crop_rgb)
                batch_tensors.append(tensor)
                valid_indices.append(i)
                
            except Exception as e:
                print(f"⚠️ Error processing crop {i}: {e}")
                continue
        
        if not batch_tensors:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        
        # Stack into batch dan move ke device
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Extract features
        with torch.no_grad():
            # Get backbone features
            backbone_features = self.backbone(batch)
            
            # Flatten jika perlu
            if len(backbone_features.shape) > 2:
                backbone_features = backbone_features.view(backbone_features.size(0), -1)
            
            # Reduce dimensions
            reduced_features = self.feature_reducer(backbone_features)
        
        # Convert ke numpy
        features = reduced_features.cpu().numpy()
        
        # Track performance
        extraction_time = time.time() - start_time
        self.extraction_times.append(extraction_time)
        self.total_extractions += len(features)
        
        # Pad hasil jika ada crop yang di-skip
        if len(valid_indices) < len(image_crops):
            full_features = np.zeros((len(image_crops), self.feature_dim), dtype=np.float32)
            full_features[valid_indices] = features
            return full_features
        
        return features
    
    def extract_single_feature(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Extract feature dari single pedestrian crop
        
        Args:
            image_crop: Single pedestrian crop (BGR format)
            
        Returns:
            Feature vector of shape [feature_dim]
        """
        features = self.extract_features([image_crop])
        return features[0] if len(features) > 0 else np.zeros(self.feature_dim, dtype=np.float32)
    
    def crop_detections(self, image: np.ndarray, detections: List[Dict]) -> List[np.ndarray]:
        """
        Crop pedestrian regions dari image berdasarkan detections
        
        Args:
            image: Full image (BGR format)
            detections: List of detection dictionaries dengan 'bbox' key
            
        Returns:
            List of cropped pedestrian images
        """
        crops = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Add small padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Validate bbox
            if x2 <= x1 or y2 <= y1:
                # Invalid bbox, create dummy crop
                crop = np.zeros((64, 32, 3), dtype=np.uint8)
                print(f"⚠️ Invalid bbox [{x1},{y1},{x2},{y2}], using dummy crop")
            else:
                # Crop region
                crop = image[y1:y2, x1:x2]
                
                # Check if crop is empty
                if crop.size == 0:
                    crop = np.zeros((64, 32, 3), dtype=np.uint8)
            
            crops.append(crop)
        
        return crops
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity antara dua feature vectors
        
        Args:
            features1: First feature vector [feature_dim]
            features2: Second feature vector [feature_dim]
            
        Returns:
            Cosine similarity score [0, 1]
        """
        # Ensure features normalized (harusnya sudah dari L2Norm layer)
        f1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
        f2_norm = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(f1_norm, f2_norm)
        
        # Clamp ke [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
    
    def compute_similarity_matrix(self, features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix antara dua sets features
        
        Args:
            features1: Feature matrix [N1, feature_dim]
            features2: Feature matrix [N2, feature_dim]
            
        Returns:
            Similarity matrix [N1, N2]
        """
        if features1.size == 0 or features2.size == 0:
            return np.zeros((features1.shape[0], features2.shape[0]), dtype=np.float32)
        
        # Normalize features
        f1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
        f2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(f1_norm, f2_norm.T)
        
        # Clamp values
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        return similarity_matrix
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.extraction_times:
            return {
                'avg_extraction_time': 0,
                'total_extractions': 0,
                'features_per_second': 0,
                'model_size_mb': self._estimate_model_size()
            }
        
        avg_time = np.mean(self.extraction_times)
        features_per_sec = self.total_extractions / sum(self.extraction_times)
        
        return {
            'avg_extraction_time': avg_time,
            'total_extractions': self.total_extractions,
            'features_per_second': features_per_sec,
            'model_size_mb': self._estimate_model_size(),
            'backbone_model': self.model_name
        }
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB"""
        total_params = sum(p.numel() for p in self.backbone.parameters())
        total_params += sum(p.numel() for p in self.feature_reducer.parameters())
        
        # Assume float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def visualize_crops(self, crops: List[np.ndarray], features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize pedestrian crops untuk debugging
        
        Args:
            crops: List of pedestrian crops
            features: Optional feature vectors untuk display similarity
            
        Returns:
            Concatenated visualization image
        """
        if not crops:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Resize semua crops ke same size untuk visualization
        vis_size = (64, 128)  # width, height
        resized_crops = []
        
        for i, crop in enumerate(crops):
            if crop.size == 0:
                resized = np.zeros((vis_size[1], vis_size[0], 3), dtype=np.uint8)
            else:
                resized = cv2.resize(crop, vis_size)
            
            # Add text label
            label = f"ID_{i}"
            if features is not None and i < len(features):
                feat_norm = np.linalg.norm(features[i])
                label += f" |f|={feat_norm:.2f}"
            
            cv2.putText(resized, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, (0, 255, 0), 1)
            resized_crops.append(resized)
        
        # Concatenate horizontally
        if len(resized_crops) == 1:
            return resized_crops[0]
        
        vis_image = np.hstack(resized_crops)
        return vis_image
    
    def get_extractor_info(self) -> Dict:
        """Get information tentang feature extractor"""
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'device': self.device,
            'image_size': self.image_size,
            'backbone_type': self.backbone.__class__.__name__,
            'backbone_output_dim': self._get_backbone_output_dim(),
            'estimated_size_mb': self._estimate_model_size(),
            'iot_optimized': 'mobilenet' in self.model_name.lower() or 'efficient' in self.model_name.lower()
        }


# L2 Normalization layer
class L2Norm(nn.Module):
    """L2 normalization layer untuk feature normalization"""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


# Backward compatibility - alias untuk nama lama
OSNetExtractor = MobileNetV3Extractor


# Example usage dan testing
if __name__ == "__main__":
    # Test MobileNetV3 extractor
    extractor = MobileNetV3Extractor(feature_dim=128)
    
    # Print model info
    info = extractor.get_extractor_info()
    print("MobileNetV3 Feature Extractor Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test dengan dummy crops
    print("\nTesting feature extraction...")
    
    # Create dummy pedestrian crops
    dummy_crops = [
        np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8),
        np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8),
        np.random.randint(0, 255, (150, 70, 3), dtype=np.uint8)
    ]
    
    # Extract features
    features = extractor.extract_features(dummy_crops)
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature norms: {[np.linalg.norm(f) for f in features]}")
    
    # Test similarity computation
    if len(features) >= 2:
        sim = extractor.compute_similarity(features[0], features[1])
        print(f"Similarity between crop 0 and 1: {sim:.4f}")
    
    # Test similarity matrix
    sim_matrix = extractor.compute_similarity_matrix(features[:2], features[1:])
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    # Performance stats
    perf_stats = extractor.get_performance_stats()
    print(f"Performance stats: {perf_stats}")