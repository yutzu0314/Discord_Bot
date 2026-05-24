# src/utils/mot17_loader.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import csv

class MOT17Loader:
    """MOT17 dataset loader for TrackGuard evaluation"""
    
    def __init__(self, dataset_root: str, sequence_name: str):
        """
        Initialize MOT17 loader
        
        Args:
            dataset_root: Path to MOT17 dataset root
            sequence_name: Name of sequence (e.g., 'MOT17-02-SDP')
        """
        self.dataset_root = Path(dataset_root)
        self.sequence_name = sequence_name
        self.sequence_path = self.dataset_root / sequence_name
        
        # Validate paths
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence not found: {self.sequence_path}")
        
        # Set up paths
        self.img_path = self.sequence_path / "img1"
        self.gt_path = self.sequence_path / "gt" / "gt.txt"
        
        if not self.img_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_path}")
        
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.gt_path}")
        
        # Load ground truth data
        self.ground_truth = self._load_ground_truth()
        
        # Get sequence info
        self.total_frames = self._get_total_frames()
        
        print(f"MOT17 Loader initialized:")
        print(f"  Sequence: {sequence_name}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  GT annotations: {len(self.ground_truth)}")
    
    def _load_ground_truth(self) -> List[Dict]:
        """
        Load ground truth annotations from gt.txt
        
        MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
        """
        annotations = []
        
        try:
            with open(self.gt_path, 'r') as f:
                csv_reader = csv.reader(f)
                for line_num, row in enumerate(csv_reader, 1):
                    if len(row) >= 9:  # MOT format has 9 columns minimum
                        try:
                            frame_id = int(row[0])
                            track_id = int(row[1])
                            bb_left = float(row[2])
                            bb_top = float(row[3])
                            bb_width = float(row[4])
                            bb_height = float(row[5])
                            conf = float(row[6])
                            class_id = int(row[7]) if row[7] != '-1' else 1  # Default to person
                            visibility = float(row[8]) if row[8] != '-1' else 1.0
                            
                            # Skip invalid bboxes
                            if bb_width <= 0 or bb_height <= 0:
                                continue
                            
                            # Convert to xyxy format
                            x1 = bb_left
                            y1 = bb_top
                            x2 = bb_left + bb_width
                            y2 = bb_top + bb_height
                            
                            annotation = {
                                'frame_id': frame_id,
                                'track_id': track_id,
                                'bbox': [x1, y1, x2, y2],  # xyxy format
                                'bbox_xywh': [bb_left, bb_top, bb_width, bb_height],  # Original format
                                'confidence': conf,
                                'class_id': class_id,
                                'visibility': visibility
                            }
                            annotations.append(annotation)
                            
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Skip line {line_num} due to parsing error: {e}")
                            continue
                    else:
                        print(f"Warning: Line {line_num} has insufficient columns: {len(row)}")
        
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        print(f"Loaded {len(annotations)} annotations from GT file")
        return annotations
    
    def _get_total_frames(self) -> int:
        """Get total number of frames in sequence"""
        if not self.ground_truth:
            # Count image files if GT is empty
            image_files = list(self.img_path.glob("*.jpg"))
            return len(image_files)
        
        # Get max frame ID from ground truth
        max_frame = max(ann['frame_id'] for ann in self.ground_truth)
        return max_frame
    
    def get_frame_image(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Load frame image by frame ID
        
        Args:
            frame_id: Frame number (1-indexed)
            
        Returns:
            Frame image or None if not found
        """
        # MOT17 uses 6-digit frame numbering (000001.jpg, 000002.jpg, etc.)
        frame_filename = f"{frame_id:06d}.jpg"
        frame_path = self.img_path / frame_filename
        
        if not frame_path.exists():
            return None
        
        try:
            image = cv2.imread(str(frame_path))
            return image
        except Exception as e:
            print(f"Error loading frame {frame_id}: {e}")
            return None
    
    def get_frame_annotations(self, frame_id: int) -> List[Dict]:
        """
        Get ground truth annotations for specific frame
        
        Args:
            frame_id: Frame number (1-indexed)
            
        Returns:
            List of annotations for the frame
        """
        frame_annotations = []
        
        for ann in self.ground_truth:
            if ann['frame_id'] == frame_id:
                frame_annotations.append(ann)
        
        return frame_annotations
    
    def get_sequence_info(self) -> Dict:
        """Get sequence information"""
        frame_counts = {}
        class_counts = {}
        
        for ann in self.ground_truth:
            frame_id = ann['frame_id']
            class_id = ann['class_id']
            
            frame_counts[frame_id] = frame_counts.get(frame_id, 0) + 1
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        return {
            'sequence_name': self.sequence_name,
            'total_frames': self.total_frames,
            'total_annotations': len(self.ground_truth),
            'avg_objects_per_frame': len(self.ground_truth) / max(1, self.total_frames),
            'unique_track_ids': len(set(ann['track_id'] for ann in self.ground_truth)),
            'class_distribution': class_counts,
            'frame_distribution': frame_counts
        }
    
    def filter_annotations_by_class(self, class_ids: List[int]) -> List[Dict]:
        """
        Filter annotations by class IDs
        
        Args:
            class_ids: List of class IDs to keep
            
        Returns:
            Filtered annotations
        """
        filtered = []
        for ann in self.ground_truth:
            if ann['class_id'] in class_ids:
                filtered.append(ann)
        
        return filtered
    
    def get_available_sequences(dataset_root: str) -> List[str]:
        """
        Get list of available MOT17 sequences
        
        Args:
            dataset_root: Path to MOT17 dataset root
            
        Returns:
            List of sequence names
        """
        dataset_path = Path(dataset_root)
        if not dataset_path.exists():
            return []
        
        sequences = []
        for item in dataset_path.iterdir():
            if item.is_dir() and item.name.startswith('MOT17-'):
                sequences.append(item.name)
        
        return sorted(sequences)
    
    def __len__(self) -> int:
        """Get total number of frames"""
        return self.total_frames
    
    def __str__(self) -> str:
        """String representation"""
        return f"MOT17Loader({self.sequence_name}, {self.total_frames} frames)"