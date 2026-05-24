"""
MOT17 Ground Truth Loader
=========================

Loader untuk membaca ground truth annotations dari dataset MOT17.
"""

import os
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class MOT17GTLoader:
    """
    Loader untuk MOT17 ground truth annotations.
    """
    
    def __init__(self, mot17_root: str, sequence_name: str):
        """
        Initialize MOT17 GT loader.
        
        Args:
            mot17_root: Root directory MOT17 (misal: C:\\Users\\phantom\\TrackGraph-SHA\\isolasi\\MOT17)
            sequence_name: Nama sequence (misal: 'MOT17-02-FRCNN', 'MOT17-05-SDP')
        """
        self.mot17_root = Path(mot17_root)
        self.sequence_name = sequence_name
        
        # Cek path GT file - coba dengan train/ dulu, kalau tidak ada coba langsung
        gt_file = self.mot17_root / "train" / sequence_name / "gt" / "gt.txt"
        
        if not gt_file.exists():
            # Coba langsung di root tanpa folder train
            gt_file = self.mot17_root / sequence_name / "gt" / "gt.txt"
        
        if not gt_file.exists():
            raise FileNotFoundError(
                f"GT file tidak ditemukan. Dicoba:\n"
                f"  1. {self.mot17_root / 'train' / sequence_name / 'gt' / 'gt.txt'}\n"
                f"  2. {gt_file}"
            )
        
        self.gt_file = gt_file
        self.gt_data = self._load_gt_file()
        
        print(f"✓ Loaded GT: {len(self.gt_data)} annotations from {gt_file}")
    
    def _load_gt_file(self) -> Dict[int, List[Dict]]:
        """
        Load GT file dan organize per frame.
        
        Format MOT17 GT: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <class>, <visibility>
        
        Returns:
            gt_data: Dict {frame_id: [annotations]}
        """
        gt_data = {}
        
        with open(self.gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 9:
                    continue
                
                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6])
                    class_id = int(parts[7])
                    visibility = float(parts[8])
                    
                    # Convert tlwh (top-left width height) ke xyxy (x1 y1 x2 y2)
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                    
                    annotation = {
                        'frame_id': frame_id,
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],  # xyxy format
                        'bbox_xyxy': [x1, y1, x2, y2],
                        'bbox_tlwh': [x, y, w, h],  # tlwh format
                        'class_id': class_id,
                        'category_id': class_id,  # Alias
                        'confidence': conf,
                        'visibility': visibility
                    }
                    
                    if frame_id not in gt_data:
                        gt_data[frame_id] = []
                    
                    gt_data[frame_id].append(annotation)
                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing GT line: {line[:50]}... Error: {e}")
                    continue
        
        return gt_data
    
    def get_frame_annotations(self, frame_id: int) -> List[Dict]:
        """
        Get annotations untuk frame tertentu.
        
        Args:
            frame_id: Frame ID (1-indexed)
            
        Returns:
            annotations: List of annotation dicts
        """
        return self.gt_data.get(frame_id, [])
    
    def get_frame_image_path(self, frame_id: int) -> Optional[Path]:
        """
        Get path ke image untuk frame tertentu.
        
        Args:
            frame_id: Frame ID (1-indexed)
            
        Returns:
            image_path: Path ke image file
        """
        # Format: MOT17-XX-YYY/img1/000001.jpg
        # Cek dengan train/ dulu, kalau tidak ada coba langsung
        img_dir = self.mot17_root / "train" / self.sequence_name / "img1"
        
        if not img_dir.exists():
            # Coba langsung di root tanpa folder train
            img_dir = self.mot17_root / self.sequence_name / "img1"
            if not img_dir.exists():
                return None
        
        # Frame ID biasanya 6-digit dengan leading zeros
        img_name = f"{frame_id:06d}.jpg"
        img_path = img_dir / img_name
        
        if img_path.exists():
            return img_path
        
        # Try alternative formats
        for ext in ['.png', '.PNG', '.jpeg', '.JPEG']:
            alt_path = img_dir / f"{frame_id:06d}{ext}"
            if alt_path.exists():
                return alt_path
        
        return None
    
    def get_num_frames(self) -> int:
        """
        Get total number of frames.
        """
        if not self.gt_data:
            return 0
        return max(self.gt_data.keys()) if self.gt_data else 0
    
    def get_all_frame_ids(self) -> List[int]:
        """
        Get list of all frame IDs.
        """
        return sorted(self.gt_data.keys())
