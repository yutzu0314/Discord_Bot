import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import csv
import os

class MOTMetrics:
    """Calculate standard MOT Challenge metrics for TrackGuard evaluation"""
    
    def __init__(self):
        # Initialize counters
        self.frame_count = 0
        self.gt_tracks = {}  # gt_id -> list of detections
        self.pred_tracks = {}  # pred_id -> list of detections
        
        # Per-frame statistics
        self.tp_count = 0  # True positives
        self.fp_count = 0  # False positives
        self.fn_count = 0  # False negatives
        self.id_switches = 0  # ID switches
        self.gt_count = 0  # Total ground truth objects
        
        # Track-level statistics
        self.matches = defaultdict(list)  # gt_id -> [(frame_idx, pred_id), ...]
        self.track_fragmentations = 0
        
        # Accumulated results
        self.frame_results = []
        
        # Mapping for current frame
        self.current_matches = {}  # gt_id -> pred_id for current frame
        self.previous_matches = {}  # gt_id -> pred_id for previous frame
        
    def update(self, gt_objects: List[Dict], pred_objects: List[Dict], 
              frame_idx: int, iou_threshold: float = 0.5) -> Dict:
        """
        Update metrics with new frame detections and ground truth
        
        Args:
            gt_objects: List of ground truth objects with bbox, track_id
            pred_objects: List of predicted objects with bbox, track_id
            frame_idx: Current frame index
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dict: Current frame metrics
        """
        self.frame_count += 1
        
        # Reset current matches
        self.previous_matches = self.current_matches.copy()
        self.current_matches = {}
        
        # Update track history
        for gt in gt_objects:
            gt_id = gt['track_id']
            if gt_id not in self.gt_tracks:
                self.gt_tracks[gt_id] = []
            self.gt_tracks[gt_id].append((frame_idx, gt))
            
        for pred in pred_objects:
            pred_id = pred['track_id']
            if pred_id not in self.pred_tracks:
                self.pred_tracks[pred_id] = []
            self.pred_tracks[pred_id].append((frame_idx, pred))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(gt_objects), len(pred_objects)))
        for i, gt in enumerate(gt_objects):
            for j, pred in enumerate(pred_objects):
                iou_matrix[i, j] = self._calculate_iou(gt['bbox'], pred['bbox'])
        
        # Match detections to ground truth
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        # Find matches above threshold, sort by IoU
        matches = []
        for i in range(len(gt_objects)):
            for j in range(len(pred_objects)):
                if iou_matrix[i, j] >= iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        # Sort matches by IoU (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Assign matches greedily
        for gt_idx, pred_idx, iou in matches:
            if gt_idx not in matched_gt_indices and pred_idx not in matched_pred_indices:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(pred_idx)
                
                # Record match
                gt_id = gt_objects[gt_idx]['track_id']
                pred_id = pred_objects[pred_idx]['track_id']
                self.matches[gt_id].append((frame_idx, pred_id))
                self.current_matches[gt_id] = pred_id
        
        # Count metrics
        tp = len(matched_gt_indices)
        fp = len(pred_objects) - tp
        fn = len(gt_objects) - tp
        
        # Check for ID switches
        id_switches = 0
        for gt_id, current_pred_id in self.current_matches.items():
            if gt_id in self.previous_matches:
                prev_pred_id = self.previous_matches[gt_id]
                if current_pred_id != prev_pred_id:
                    id_switches += 1
        
        # Accumulate results
        self.tp_count += tp
        self.fp_count += fp
        self.fn_count += fn
        self.id_switches += id_switches
        self.gt_count += len(gt_objects)
        
        # Calculate fragmentations
        self._update_fragmentations()
        
        # Return current frame metrics
        frame_metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'id_switches': id_switches,
            'mota': self._calculate_mota(tp, fp, fn, id_switches, len(gt_objects)),
            'precision': self._calculate_precision(tp, fp),
            'recall': self._calculate_recall(tp, fn)
        }
        
        self.frame_results.append(frame_metrics)
        
        return frame_metrics
    
    def _update_fragmentations(self):
        """Calculate track fragmentations (number of times tracks are interrupted)"""
        # Reset count
        self.track_fragmentations = 0
        
        # For each GT track, count discontinuities in tracking
        for gt_id, match_history in self.matches.items():
            # Sort by frame index
            sorted_matches = sorted(match_history, key=lambda x: x[0])
            
            # Count gaps in frame indices
            prev_frame = None
            for frame_idx, _ in sorted_matches:
                if prev_frame is not None and frame_idx > prev_frame + 1:
                    self.track_fragmentations += 1
                prev_frame = frame_idx
    
    def _calculate_iou(self, bbox1, bbox2) -> float:
        """Calculate IoU between two bounding boxes"""
        # Convert to [x1,y1,x2,y2] format if needed
        if len(bbox1) == 4 and len(bbox2) == 4:
            x1, y1, x2, y2 = bbox1
            x1_, y1_, x2_, y2_ = bbox2
        else:
            return 0.0
        
        # Calculate intersection
        x_left = max(x1, x1_)
        y_top = max(y1, y1_)
        x_right = min(x2, x2_)
        y_bottom = min(y2, y2_)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x2_ - x1_) * (y2_ - y1_)
        
        # Calculate IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
    
    def _calculate_mota(self, tp, fp, fn, id_switches, n_gt) -> float:
        """Calculate MOTA (Multiple Object Tracking Accuracy)"""
        if n_gt == 0:
            return 1.0  # Perfect score for empty ground truth
        
        return 1.0 - float(fp + fn + id_switches) / max(1, n_gt)
    
    def _calculate_precision(self, tp, fp) -> float:
        """Calculate precision"""
        if tp + fp == 0:
            return 1.0  # No detections, return perfect precision
        
        return float(tp) / (tp + fp)
    
    def _calculate_recall(self, tp, fn) -> float:
        """Calculate recall"""
        if tp + fn == 0:
            return 1.0  # No ground truth, return perfect recall
        
        return float(tp) / (tp + fn)
    
    def get_summary(self) -> Dict:
        """Calculate summary metrics"""
        # Overall MOTA
        mota = 1.0 - float(self.fp_count + self.fn_count + self.id_switches) / max(1, self.gt_count)
        
        # Precision
        precision = float(self.tp_count) / max(1, self.tp_count + self.fp_count)
        
        # Recall
        recall = float(self.tp_count) / max(1, self.tp_count + self.fn_count)
        
        # Calculate IDF1 (ID F1 Score)
        idf1, idp, idr = self._calculate_idf1()
        
        # Mostly tracked (MT), Partly tracked (PT), Mostly lost (ML)
        mt, pt, ml = self._calculate_tracking_ratios()
        
        # Fragmentation ratio
        frag_ratio = float(self.track_fragmentations) / max(1, len(self.gt_tracks))
        
        return {
            'mota': mota,
            'motp': self._calculate_motp(),
            'precision': precision,
            'recall': recall,
            'id_switches': self.id_switches,
            'fragmentations': self.track_fragmentations,
            'idf1': idf1,
            'idp': idp,
            'idr': idr,
            'mt_ratio': mt,
            'pt_ratio': pt,
            'ml_ratio': ml,
            'fragmentation_ratio': frag_ratio,
            'fp': self.fp_count,
            'fn': self.fn_count,
            'tp': self.tp_count,
            'total_gt': self.gt_count,
            'num_frames': self.frame_count
        }
    
    def _calculate_motp(self) -> float:
        """Calculate MOTP (Multiple Object Tracking Precision)"""
        # MOTP is the average IoU across all matches
        total_iou = 0.0
        match_count = 0
        
        for gt_id, match_history in self.matches.items():
            for frame_idx, pred_id in match_history:
                # Find the corresponding detections
                gt_detection = None
                pred_detection = None
                
                for f_idx, det in self.gt_tracks[gt_id]:
                    if f_idx == frame_idx:
                        gt_detection = det
                        break
                
                for f_idx, det in self.pred_tracks[pred_id]:
                    if f_idx == frame_idx:
                        pred_detection = det
                        break
                
                if gt_detection and pred_detection:
                    iou = self._calculate_iou(gt_detection['bbox'], pred_detection['bbox'])
                    total_iou += iou
                    match_count += 1
        
        if match_count == 0:
            return 0.0
            
        return total_iou / match_count
    
    def _calculate_idf1(self) -> Tuple[float, float, float]:
        """Calculate IDF1 (ID F1 Score), IDP (ID Precision), and IDR (ID Recall)"""
        # Calculate ID global association
        id_tp = 0  # Correctly identified detections
        id_fp = 0  # False positive IDs
        id_fn = 0  # False negative IDs
        
        # For each ground truth trajectory
        for gt_id, gt_dets in self.gt_tracks.items():
            # Count number of frames this ground truth object appears
            gt_frames = set(frame_idx for frame_idx, _ in gt_dets)
            
            # Count matched frames
            if gt_id in self.matches:
                matched_frames = set(frame_idx for frame_idx, _ in self.matches[gt_id])
                id_tp += len(matched_frames)
                id_fn += len(gt_frames) - len(matched_frames)
            else:
                id_fn += len(gt_frames)
        
        # Count false positives
        for pred_id, pred_dets in self.pred_tracks.items():
            pred_frames = set(frame_idx for frame_idx, _ in pred_dets)
            
            # Count frames where this prediction was matched to any GT
            matched_frames = set()
            for gt_id, matches in self.matches.items():
                for frame_idx, p_id in matches:
                    if p_id == pred_id:
                        matched_frames.add(frame_idx)
            
            id_fp += len(pred_frames) - len(matched_frames)
        
        # Calculate IDP, IDR, IDF1
        idp = float(id_tp) / max(1, id_tp + id_fp)
        idr = float(id_tp) / max(1, id_tp + id_fn)
        
        if idp + idr == 0:
            idf1 = 0.0
        else:
            idf1 = 2 * idp * idr / (idp + idr)
        
        return idf1, idp, idr
    
    def _calculate_tracking_ratios(self) -> Tuple[float, float, float]:
        """Calculate Mostly Tracked (MT), Partly Tracked (PT), and Mostly Lost (ML) ratios"""
        if not self.gt_tracks:
            return 0.0, 0.0, 0.0
            
        mt_count = 0  # Mostly tracked (>= 80%)
        pt_count = 0  # Partly tracked (20% - 80%)
        ml_count = 0  # Mostly lost (< 20%)
        
        for gt_id, gt_dets in self.gt_tracks.items():
            gt_frames = set(frame_idx for frame_idx, _ in gt_dets)
            total_gt_frames = len(gt_frames)
            
            # Count matched frames
            matched_frames = set()
            if gt_id in self.matches:
                matched_frames = set(frame_idx for frame_idx, _ in self.matches[gt_id])
            
            tracking_ratio = len(matched_frames) / max(1, total_gt_frames)
            
            if tracking_ratio >= 0.8:
                mt_count += 1
            elif tracking_ratio >= 0.2:
                pt_count += 1
            else:
                ml_count += 1
        
        total_tracks = len(self.gt_tracks)
        
        return (
            float(mt_count) / total_tracks,
            float(pt_count) / total_tracks,
            float(ml_count) / total_tracks
        )
    
    def export_to_csv(self, file_path: str) -> None:
        """Export metrics to CSV file"""
        summary = self.get_summary()
        
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            
            for key, value in summary.items():
                writer.writerow([key, f"{value:.6f}" if isinstance(value, float) else value])
    
    def export_per_frame_metrics(self, file_path: str) -> None:
        """Export per-frame metrics to CSV file"""
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'TP', 'FP', 'FN', 'ID_Switches', 'MOTA', 'Precision', 'Recall'])
            
            for i, metrics in enumerate(self.frame_results):
                writer.writerow([
                    i+1,
                    metrics['tp'],
                    metrics['fp'],
                    metrics['fn'],
                    metrics['id_switches'],
                    f"{metrics['mota']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}"
                ])
    
    def export_track_analysis(self, file_path: str) -> None:
        """Export per-track analysis to CSV file"""
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['GT_ID', 'Total_Frames', 'Matched_Frames', 'Coverage', 'Fragmentation', 'ID_Changes'])
            
            for gt_id, gt_dets in self.gt_tracks.items():
                gt_frames = set(frame_idx for frame_idx, _ in gt_dets)
                total_frames = len(gt_frames)
                
                # Count matched frames and fragmentations
                matched_frames = set()
                fragmentations = 0
                id_changes = 0
                prev_pred_id = None
                
                if gt_id in self.matches:
                    matches = sorted(self.matches[gt_id], key=lambda x: x[0])  # Sort by frame_idx
                    
                    # Count matched frames
                    for frame_idx, _ in matches:
                        matched_frames.add(frame_idx)
                    
                    # Count ID changes
                    for _, pred_id in matches:
                        if prev_pred_id is not None and prev_pred_id != pred_id:
                            id_changes += 1
                        prev_pred_id = pred_id
                    
                    # Count fragmentations (track interruptions)
                    prev_frame = None
                    for frame_idx, _ in matches:
                        if prev_frame is not None and frame_idx > prev_frame + 1:
                            fragmentations += 1
                        prev_frame = frame_idx
                
                coverage = len(matched_frames) / total_frames if total_frames > 0 else 0.0
                
                writer.writerow([
                    gt_id,
                    total_frames,
                    len(matched_frames),
                    f"{coverage:.6f}",
                    fragmentations,
                    id_changes
                ])
    
    def export_all_metrics(self, output_dir: str) -> Dict[str, str]:
        """Export all metrics to the specified directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export summary metrics
        summary_path = os.path.join(output_dir, 'mot_summary_metrics.csv')
        self.export_to_csv(summary_path)
        
        # Export per-frame metrics
        frame_path = os.path.join(output_dir, 'mot_frame_metrics.csv')
        self.export_per_frame_metrics(frame_path)
        
        # Export track analysis
        track_path = os.path.join(output_dir, 'mot_track_analysis.csv')
        self.export_track_analysis(track_path)
        
        return {
            'summary': summary_path,
            'frame': frame_path,
            'track': track_path
        }
    
    def read_mot_ground_truth(self, gt_file_path: str) -> Dict[int, List[Dict]]:
        """
        Read MOT Challenge ground truth file
        
        Args:
            gt_file_path: Path to gt.txt file
            
        Returns:
            Dict: Frame index -> List of ground truth objects
        """
        ground_truth = defaultdict(list)
        
        with open(gt_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                    
                frame_idx = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(parts[7]) if len(parts) > 7 else 1  # Default to class 1 (often "person")
                
                # Convert to [x1,y1,x2,y2] format
                bbox = [x, y, x+w, y+h]
                
                ground_truth[frame_idx].append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': class_id
                })
        
        return ground_truth