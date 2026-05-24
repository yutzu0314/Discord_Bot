"""
Standalone Simple MOTA Calculator
==================================

Standalone version of SimpleMOTACalculator for ByteTrack evaluation.
Extracted from mot_evaluator_hungarian.py without external dependencies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import scipy.optimize


class SimpleMOTACalculator:
    """
    Simple MOTA Calculator - standalone version
    Calculates MOTA, IDF1, Precision, Recall and detailed metrics.
    """
    
    def __init__(self):
        # MOT Challenge classes yang dievaluasi
        self.MOT_EVAL_CLASSES = {1, 2, 7}  # person, person_on_vehicle, static_person
        
        # Fine-tuned detector classes
        self.DETECTOR_CLASSES = {1, 2, 7}  # Classes yang di-detect oleh fine-tuned model
        
        # Mapping dari detector class ke MOT eval class
        self.CLASS_MAPPING = {
            1: 1,  # person → person
            2: 2,  # person_on_vehicle → person_on_vehicle  
            7: 7   # static_person → static_person
        }
        
        self.reset()
    
    def reset(self):
        """Reset semua counter"""
        # Core MOTA components
        self.total_gt = 0
        self.total_fp = 0  # False Positives
        self.total_fn = 0  # False Negatives (missed)
        self.total_id_switches = 0
        
        # IDF1 components
        self.total_idtp = 0  # Identity True Positives
        self.total_idfp = 0  # Identity False Positives  
        self.total_idfn = 0  # Identity False Negatives
        
        # Tracking untuk ID switches dan IDF1
        self.prev_matches = {}  # {track_id: gt_id}
        self.gt_track_assignments = defaultdict(list)  # {gt_id: [track_ids]}
        self.track_gt_assignments = defaultdict(list)  # {track_id: [gt_ids]}
        
        # Frame details
        self.frame_results = []
        
        # Statistics
        self.total_frames = 0
        self.total_tracks_created = 0
        
        # Class distribution tracking
        self.gt_class_distribution = defaultdict(int)
        self.det_class_distribution = defaultdict(int)
        
        print("✓ MOTA Calculator reset (Standalone version)")
        print(f"  Detector classes: {self.DETECTOR_CLASSES}")
        print(f"  Evaluation classes: {self.MOT_EVAL_CLASSES}")
        print(f"  IDF1 calculation: ENABLED")
    
    def filter_gt_annotations(self, gt_annotations: List[Dict]) -> List[Dict]:
        """Filter GT annotations untuk MOT Challenge classes."""
        filtered = []
        for ann in gt_annotations:
            class_id = ann.get('class_id', ann.get('category_id', -1))
            if class_id in self.MOT_EVAL_CLASSES:
                filtered.append(ann)
                self.gt_class_distribution[class_id] += 1
        return filtered
    
    def filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections berdasarkan class dan confidence."""
        filtered = []
        for det in detections:
            # Handle both dict and object formats
            if isinstance(det, dict):
                class_id = det.get('class_id', det.get('category_id', -1))
                conf = det.get('confidence', det.get('conf', 0.0))
            else:
                # Object with current_detection attribute (current_detection is a dict)
                if hasattr(det, 'current_detection'):
                    if isinstance(det.current_detection, dict):
                        class_id = det.current_detection.get('class_id', -1)
                        conf = det.current_detection.get('confidence', 0.0)
                    else:
                        # current_detection is an object
                        class_id = getattr(det.current_detection, 'class_id', -1)
                        conf = getattr(det.current_detection, 'confidence', 0.0)
                else:
                    # Try to get class_id directly from object
                    class_id = getattr(det, 'class_id', getattr(det, 'category_id', -1))
                    conf = getattr(det, 'confidence', getattr(det, 'conf', 0.0))
            
            # If class_id still not found, default to 1 (person for MOT17)
            if class_id == -1:
                class_id = 1
            
            if class_id in self.DETECTOR_CLASSES and conf > 0.0:
                filtered.append(det)
                self.det_class_distribution[class_id] += 1
        
        return filtered
    
    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bboxes in xyxy format."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def extract_bbox(self, obj: Dict) -> Optional[List[float]]:
        """Extract bbox from object (dict or object with attributes)."""
        if isinstance(obj, dict):
            if 'bbox' in obj:
                return obj['bbox']
            elif 'bbox_xyxy' in obj:
                return obj['bbox_xyxy']
        else:
            if hasattr(obj, 'current_detection'):
                det = obj.current_detection
                if isinstance(det, dict):
                    return det.get('bbox', det.get('bbox_xyxy'))
                elif hasattr(det, 'bbox'):
                    return det.bbox
            elif hasattr(obj, 'bbox'):
                return obj.bbox
        return None
    
    def extract_track_id(self, obj: Dict) -> Optional[int]:
        """Extract track_id from object."""
        if isinstance(obj, dict):
            return obj.get('track_id')
        else:
            return getattr(obj, 'track_id', None)
    
    def update_frame(self, gt_annotations: List[Dict], tracking_results: List[Dict], frame_id: int):
        """
        Update MOTA calculation untuk satu frame.
        
        Args:
            gt_annotations: List of GT annotations (dicts with 'bbox', 'track_id', 'class_id')
            tracking_results: List of tracking results (dicts or objects with 'track_id', 'bbox', 'confidence')
            frame_id: Frame ID
        """
        # Filter GT and detections
        filtered_gt = self.filter_gt_annotations(gt_annotations)
        filtered_det = self.filter_detections(tracking_results)
        
        self.total_gt += len(filtered_gt)
        self.total_frames += 1
        
        if len(filtered_gt) == 0 and len(filtered_det) == 0:
            self.frame_results.append({
                'frame_id': frame_id,
                'num_detections': 0,
                'num_matches': 0,
                'num_fp': 0,
                'num_fn': 0
            })
            return
        
        # Build cost matrix for matching
        if len(filtered_gt) > 0 and len(filtered_det) > 0:
            cost_matrix = np.ones((len(filtered_gt), len(filtered_det))) * 1e6
            
            for i, gt_ann in enumerate(filtered_gt):
                gt_bbox = gt_ann.get('bbox', gt_ann.get('bbox_xyxy'))
                if gt_bbox is None:
                    continue
                
                for j, det in enumerate(filtered_det):
                    det_bbox = self.extract_bbox(det)
                    if det_bbox is None:
                        continue
                    
                    iou = self.compute_iou(gt_bbox, det_bbox)
                    cost_matrix[i, j] = 1.0 - iou  # Convert to cost (1 - iou)
            
            # Hungarian matching
            row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
            
            # Filter matches by IoU threshold (0.5)
            matches = []
            for i, j in zip(row_indices, col_indices):
                cost = cost_matrix[i, j]
                iou = 1.0 - cost
                if iou >= 0.5:  # IoU threshold
                    matches.append((i, j, iou))
            
            # Count matches, FP, FN
            num_matches = len(matches)
            num_fp = len(filtered_det) - num_matches
            num_fn = len(filtered_gt) - num_matches
            
            self.total_fp += num_fp
            self.total_fn += num_fn
            
            # Track ID switches
            current_matches = {}
            for i, j, iou in matches:
                gt_ann = filtered_gt[i]
                det = filtered_det[j]
                
                gt_id = gt_ann.get('track_id')
                track_id = self.extract_track_id(det)
                
                if gt_id is not None and track_id is not None:
                    current_matches[track_id] = gt_id
                    
                    # Check for ID switch
                    if track_id in self.prev_matches:
                        if self.prev_matches[track_id] != gt_id:
                            self.total_id_switches += 1
                    
                    # Update IDF1 tracking
                    self.gt_track_assignments[gt_id].append(track_id)
                    self.track_gt_assignments[track_id].append(gt_id)
            
            self.prev_matches = current_matches
            
        else:
            # No GT or no detections
            if len(filtered_gt) == 0:
                num_fp = len(filtered_det)
                self.total_fp += num_fp
                num_matches = 0
                num_fn = 0
            else:  # len(filtered_det) == 0
                num_fn = len(filtered_gt)
                self.total_fn += num_fn
                num_matches = 0
                num_fp = 0
        
        self.frame_results.append({
            'frame_id': frame_id,
            'num_detections': len(filtered_det),
            'num_matches': num_matches,
            'num_fp': num_fp,
            'num_fn': num_fn
        })
    
    def compute_idf1_components(self) -> Tuple[int, int, int]:
        """Compute IDF1 components."""
        idtp = 0
        idfp = 0
        idfn = 0
        
        # For each GT track, count correct ID assignments
        for gt_id, track_ids in self.gt_track_assignments.items():
            if len(track_ids) > 0:
                # Most frequent track_id for this GT
                most_freq_track = max(set(track_ids), key=track_ids.count)
                idtp += track_ids.count(most_freq_track)
                idfp += len(track_ids) - track_ids.count(most_freq_track)
            else:
                idfn += 1
        
        # For each track, count correct GT assignments
        for track_id, gt_ids in self.track_gt_assignments.items():
            if len(gt_ids) > 0:
                most_freq_gt = max(set(gt_ids), key=gt_ids.count)
                # Already counted in above loop, so skip
        
        return idtp, idfp, idfn
    
    def get_idf1(self) -> float:
        """Calculate IDF1 score."""
        idtp, idfp, idfn = self.compute_idf1_components()
        
        if idtp + idfp == 0:
            return 0.0
        
        precision = idtp / (idtp + idfp)
        recall = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        idf1 = 2 * (precision * recall) / (precision + recall)
        return idf1
    
    def get_mota(self) -> float:
        """Calculate MOTA score."""
        if self.total_gt == 0:
            return 0.0
        
        mota = 1.0 - (self.total_fp + self.total_fn + self.total_id_switches) / self.total_gt
        return max(0.0, mota)
    
    def get_summary(self) -> Dict:
        """Get evaluation summary with MOTA, IDF1, Precision, Recall."""
        mota = self.get_mota()
        idf1 = self.get_idf1()
        
        precision = 0.0
        recall = 0.0
        
        total_detections = sum(fr['num_detections'] for fr in self.frame_results)
        total_matches = sum(fr['num_matches'] for fr in self.frame_results)
        
        if total_detections > 0:
            precision = total_matches / total_detections
        
        if self.total_gt > 0:
            recall = total_matches / self.total_gt
        
        # Get IDF1 components for detailed analysis
        idtp, idfp, idfn = self.compute_idf1_components()
        
        return {
            'MOTA': mota,
            'IDF1': idf1,
            'precision': precision,
            'recall': recall,
            'total_gt': self.total_gt,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
            'total_id_switches': self.total_id_switches,
            'total_frames': self.total_frames,
            'avg_gt_per_frame': self.total_gt / max(1, self.total_frames),
            'avg_fp_per_frame': self.total_fp / max(1, self.total_frames),
            'avg_fn_per_frame': self.total_fn / max(1, self.total_frames),
            # IDF1 detailed components
            'idtp': idtp,
            'idfp': idfp, 
            'idfn': idfn,
            'identity_precision': idtp / max(1, idtp + idfp),
            'identity_recall': idtp / max(1, idtp + idfn)
        }
