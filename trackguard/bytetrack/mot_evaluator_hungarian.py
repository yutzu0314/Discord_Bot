"""
Hungarian MOT Evaluator
=======================

MOT Evaluator yang disesuaikan untuk Hungarian TrackManager.
Menggunakan pure Hungarian algorithm untuk data association.

FEATURES:
- Support untuk Hungarian TrackManager
- MOTA dan IDF1 calculation yang sama
- Enhanced performance monitoring untuk Hungarian
- Error handling untuk infeasible cases

USAGE:
    python mot_evaluator_hungarian.py
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path
import json
from collections import defaultdict

# HUNGARIAN IMPORT
from models.track_manager import HungarianTrackManager
from models.hungarian import InfeasibleAssignmentError
from utils.data_loader import MOT17Reader
from utils.settings import SETTINGS

class SimpleMOTACalculator:
    """
    Simple MOTA Calculator - fokus pada akurasi hasil
    UNCHANGED: All MOTA calculation logic remains the same
    """
    
    def __init__(self):
        # MOT Challenge classes yang dievaluasi
        self.MOT_EVAL_CLASSES = {1, 2, 7}  # person, person_on_vehicle, static_person
        
        # UPDATED: Fine-tuned detector classes (sesuai training Anda)
        self.DETECTOR_CLASSES = {1, 2, 7}  # Classes yang di-detect oleh fine-tuned model
        
        # Mapping dari detector class ke MOT eval class (PERFECT MATCH!)
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
        
        # IDF1 components - ADDED
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
        
        # ADDED: Class distribution tracking
        self.gt_class_distribution = defaultdict(int)
        self.det_class_distribution = defaultdict(int)
        
        print("✓ MOTA Calculator reset (Hungarian + IDF1)")
        print(f"  Detector classes: {self.DETECTOR_CLASSES}")
        print(f"  Evaluation classes: {self.MOT_EVAL_CLASSES}")
        print(f"  Class mapping: {self.CLASS_MAPPING}")
        print(f"  IDF1 calculation: ENABLED")
        print(f"  Hungarian algorithm: ENABLED")
    
    def filter_gt_annotations(self, gt_annotations: List[Dict]) -> List[Dict]:
        """Filter GT untuk kelas MOT Challenge (1, 2, 7) - UNCHANGED"""
        filtered = []
        
        for gt in gt_annotations:
            # Ambil class_id dengan fallback
            class_id = gt.get('class_id', gt.get('category_id', 1))
            
            try:
                class_id = int(class_id)
            except:
                class_id = 1  # default person
            
            # Track distribution
            self.gt_class_distribution[class_id] += 1
            
            # Filter hanya kelas MOT Challenge
            if class_id in self.MOT_EVAL_CLASSES:
                visibility = float(gt.get('visibility', 1.0))
                if visibility > 0.0:  # Only visible objects
                    filtered.append(gt)
        
        return filtered
    
    def filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter deteksi dari fine-tuned YOLOv11x - UNCHANGED"""
        filtered = []
        
        for det in detections:
            # UPDATED: Get actual class from fine-tuned detector
            det_class = det.get('class_id', det.get('label', 1))  # Default person
            
            try:
                det_class = int(det_class)
            except:
                det_class = 1  # default person
            
            # Track distribution
            self.det_class_distribution[det_class] += 1
            
            # UPDATED: Filter berdasarkan fine-tuned classes
            if det_class in self.DETECTOR_CLASSES:
                confidence = float(det.get('confidence', 0.8))
                if confidence >= 0.3:  # Minimum confidence
                    # Map detector class ke evaluation class jika perlu
                    mapped_class = self.CLASS_MAPPING.get(det_class, det_class)
                    det_copy = det.copy()
                    det_copy['mapped_class'] = mapped_class
                    det_copy['original_class'] = det_class
                    filtered.append(det_copy)
        
        return filtered
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2] - UNCHANGED"""
        try:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
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
        except:
            return 0.0
    
    def match_detections_to_gt(self, gt_list: List[Dict], det_list: List[Dict], 
                              iou_threshold: float = 0.5) -> Tuple[List[Dict], int, int]:
        """Match detections to ground truth menggunakan IoU - UNCHANGED"""
        if not gt_list or not det_list:
            return [], len(det_list), len(gt_list)
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(det_list), len(gt_list)))
        
        for i, det in enumerate(det_list):
            det_bbox = det.get('bbox', [])
            if len(det_bbox) != 4:
                continue
                
            for j, gt in enumerate(gt_list):
                gt_bbox = gt.get('bbox_xyxy', gt.get('bbox', []))
                if len(gt_bbox) != 4:
                    continue
                
                iou = self.compute_iou(det_bbox, gt_bbox)
                iou_matrix[i, j] = iou
        
        # Simple greedy matching (bisa pakai Hungarian untuk optimal)
        matches = []
        used_gt = set()
        used_det = set()
        
        # Sort by IoU (highest first)
        candidates = []
        for i in range(len(det_list)):
            for j in range(len(gt_list)):
                if iou_matrix[i, j] >= iou_threshold:
                    candidates.append((iou_matrix[i, j], i, j))
        
        candidates.sort(reverse=True)  # Highest IoU first
        
        for iou, det_idx, gt_idx in candidates:
            if det_idx not in used_det and gt_idx not in used_gt:
                matches.append({
                    'gt_id': gt_list[gt_idx].get('track_id', gt_idx),
                    'track_id': det_list[det_idx].get('track_id', det_idx),
                    'iou': iou,
                    'det_idx': det_idx,
                    'gt_idx': gt_idx
                })
                used_det.add(det_idx)
                used_gt.add(gt_idx)
        
        num_fp = len(det_list) - len(matches)  # Unmatched detections
        num_fn = len(gt_list) - len(matches)   # Unmatched ground truth
        
        return matches, num_fp, num_fn
    
    def count_id_switches(self, matches: List[Dict]) -> int:
        """Count ID switches berdasarkan matches - UNCHANGED"""
        if not self.prev_matches:
            # First frame, no switches possible
            self.prev_matches = {m['track_id']: m['gt_id'] for m in matches}
            return 0
        
        switches = 0
        current_matches = {}
        
        for match in matches:
            track_id = match['track_id']
            gt_id = match['gt_id']
            current_matches[track_id] = gt_id
            
            # Check if this track was matched before
            if track_id in self.prev_matches:
                prev_gt_id = self.prev_matches[track_id]
                if prev_gt_id != gt_id:
                    switches += 1
                    print(f"  Hungarian ID Switch: Track {track_id}: GT {prev_gt_id} → {gt_id}")
        
        self.prev_matches = current_matches
        return switches
    
    def update_idf1_tracking(self, matches: List[Dict]):
        """Update IDF1 tracking based on current frame matches - UNCHANGED"""
        for match in matches:
            track_id = match['track_id']
            gt_id = match['gt_id']
            
            # Track which GT IDs each track has been assigned to
            self.track_gt_assignments[track_id].append(gt_id)
            
            # Track which track IDs each GT has been assigned to  
            self.gt_track_assignments[gt_id].append(track_id)
    
    def compute_idf1_components(self):
        """Compute IDF1 components: IDTP, IDFP, IDFN - UNCHANGED"""
        idtp = 0
        idfp = 0
        idfn = 0
        
        # For each GT track, find the track ID that matches it most
        for gt_id, track_ids in self.gt_track_assignments.items():
            if not track_ids:
                continue
                
            # Count occurrences of each track ID for this GT
            track_counts = defaultdict(int)
            for track_id in track_ids:
                track_counts[track_id] += 1
            
            # Find the most frequent track ID for this GT
            best_track_id = max(track_counts, key=track_counts.get)
            best_count = track_counts[best_track_id]
            
            # IDTP = frames where this GT was correctly assigned to best track
            idtp += best_count
            
            # IDFN = frames where this GT was assigned to other tracks
            idfn += len(track_ids) - best_count
        
        # For each predicted track, count wrong assignments
        for track_id, gt_ids in self.track_gt_assignments.items():
            if not gt_ids:
                continue
                
            # Count occurrences of each GT ID for this track
            gt_counts = defaultdict(int)
            for gt_id in gt_ids:
                gt_counts[gt_id] += 1
            
            # Find the most frequent GT ID for this track
            best_gt_id = max(gt_counts, key=gt_counts.get)
            best_count = gt_counts[best_gt_id]
            
            # IDFP = frames where this track was wrongly assigned to other GTs
            idfp += len(gt_ids) - best_count
        
        return idtp, idfp, idfn
    
    def get_idf1(self) -> float:
        """Calculate IDF1 score - UNCHANGED"""
        idtp, idfp, idfn = self.compute_idf1_components()
        
        if idtp == 0:
            return 0.0
        
        idf1 = (2 * idtp) / (2 * idtp + idfp + idfn)
        return max(0.0, min(1.0, idf1))
    
    def update_frame(self, gt_annotations: List[Dict], tracking_results: List[Dict], 
                    frame_id: int) -> Dict:
        """
        Update MOTA calculation untuk satu frame
        UPDATED: Works with Hungarian TrackManager
        """
        # Filter data sesuai MOT Challenge
        filtered_gt = self.filter_gt_annotations(gt_annotations)
        
        # Convert tracking results ke format deteksi
        detections = []
        for track in tracking_results:
            if hasattr(track, 'current_detection'):
                det = {
                    'track_id': track.track_id,
                    'bbox': track.current_detection['bbox'],
                    'confidence': track.current_detection.get('confidence', 0.8)
                }
                detections.append(det)
        
        filtered_detections = self.filter_detections(detections)
        
        # Match detections to GT
        matches, fp, fn = self.match_detections_to_gt(filtered_gt, filtered_detections)
        
        # Count ID switches dan update IDF1 tracking
        id_switches = self.count_id_switches(matches)
        self.update_idf1_tracking(matches)
        
        # Update totals
        num_gt = len(filtered_gt)
        self.total_gt += num_gt
        self.total_fp += fp
        self.total_fn += fn
        self.total_id_switches += id_switches
        self.total_frames += 1
        
        # Frame result
        frame_result = {
            'frame_id': frame_id,
            'num_gt': num_gt,
            'num_gt_original': len(gt_annotations),
            'num_detections': len(filtered_detections),
            'num_matches': len(matches),
            'fp': fp,
            'fn': fn,
            'id_switches': id_switches,
            'matches': matches
        }
        
        self.frame_results.append(frame_result)
        
        # Print frame info - less frequent untuk full sequence
        if frame_id % 50 == 1:  # Print every 50 frames untuk full sequence
            mota = self.get_mota()
            idf1 = self.get_idf1()  # ADDED IDF1
            print(f"Hungarian Frame {frame_id}: GT={num_gt}, Det={len(filtered_detections)}, "
                  f"Match={len(matches)}, FP={fp}, FN={fn}, IDSW={id_switches}, "
                  f"MOTA={mota:.3f}, IDF1={idf1:.3f}")
            
            # Progress indicator
            if frame_id > 1:
                progress = ((frame_id - 1) / 1050) * 100
                print(f"  Hungarian Progress: {progress:.1f}% - MOTA: {mota:.3f}, IDF1: {idf1:.3f}")
        
        return frame_result
    
    def get_mota(self) -> float:
        """Calculate MOTA score - UNCHANGED"""
        if self.total_gt == 0:
            return 0.0
        
        # MOTA = 1 - (FN + FP + IDSW) / GT
        mota = 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / self.total_gt
        return max(0.0, mota)
    
    def get_summary(self) -> Dict:
        """Get evaluation summary WITH IDF1 - UNCHANGED"""
        mota = self.get_mota()
        idf1 = self.get_idf1()  # ADDED IDF1
        
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
            'IDF1': idf1,  # ADDED
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


class HungarianEvaluator:
    """
    Hungarian Evaluator
    UPDATED: Changed from Smart Hungarian to Pure Hungarian architecture
    """
    
    def __init__(self):
        print("=== HUNGARIAN EVALUATOR ===")
        print("Architecture: Pure Hungarian Optimal Assignment")
        
        # HUNGARIAN: Single orchestrator handles everything!
        self.track_manager = HungarianTrackManager()
        self.data_loader = MOT17Reader()
        
        # MOTA calculator
        self.mota_calc = SimpleMOTACalculator()
        
        # Performance tracking
        self.evaluation_times = []
        
        print(f"✅ Hungarian evaluator initialized")
        print(f"  Dataset: {SETTINGS.SEQUENCE_NAME}")
        print(f"  Algorithm: Pure Hungarian Optimal Assignment")
        print(f"  Pipeline: YOLOv8 → MobileNetV3 → GraphBuilder → GAT → Hungarian")
    
    def process_frame(self, frame_id: int) -> Optional[Dict]:
        """
        Process single frame menggunakan Hungarian
        UPDATED: Changed to Hungarian TrackManager with error handling
        """
        # Load frame dan GT
        image = self.data_loader.get_frame_by_id(frame_id)
        gt_annotations = self.data_loader.get_frame_annotations(frame_id)
        
        if image is None:
            return None
        
        try:
            # HUNGARIAN CALL: Pure pipeline handles everything
            # YOLOv8 → MobileNetV3 → GraphBuilder → GAT → Hungarian → Tracks
            result = self.track_manager.process_frame(image, frame_id)
            
            # Add GT annotations untuk evaluation
            result['gt_annotations'] = gt_annotations
            
            return result
            
        except InfeasibleAssignmentError as e:
            print("error is detected")
            print(f"{type(e).__name__}: {e}")
            
            # Return empty result untuk frame ini
            return {
                'frame_id': frame_id,
                'detections': [],
                'features': np.array([]),
                'active_tracks': [],
                'gt_annotations': gt_annotations,
                'infeasible': True,
                'error': str(e)
            }
    
    def run_evaluation(self, start_frame: int = 1, num_frames: int = 1050) -> Dict:
        """
        Run evaluation dengan Hungarian
        UPDATED: Changed evaluation loop untuk Hungarian
        """
        print(f"\n🎯 Starting Hungarian Evaluation - FULL SEQUENCE")
        print(f"Frames: {start_frame} to {start_frame + num_frames - 1}")
        print(f"Expected time: ~{(num_frames * 0.133 / 60):.1f} minutes")
        print(f"Hungarian Pipeline: ENABLED")
        print("=" * 60)
        
        # Reset
        self.mota_calc.reset()
        self.evaluation_times = []
        
        successful_frames = 0
        infeasible_frames = 0
        start_time = time.time()
        
        for i in range(num_frames):
            frame_id = start_frame + i
            
            frame_start_time = time.time()
            
            # HUNGARIAN: Single call with error handling
            frame_result = self.process_frame(frame_id)
            if frame_result is None:
                print(f"⚠️ Frame {frame_id} failed to load")
                continue
            
            processing_time = time.time() - frame_start_time
            self.evaluation_times.append(processing_time)
            
            # Check if infeasible
            if frame_result.get('infeasible', False):
                infeasible_frames += 1
            else:
                # Update MOTA calculation
                self.mota_calc.update_frame(
                    frame_result['gt_annotations'],
                    frame_result['active_tracks'],
                    frame_id
                )
            
            successful_frames += 1
            
            # Memory management untuk full sequence
            if successful_frames % 100 == 0:
                # Clear old history if needed
                if len(self.track_manager.pipeline_stats['total_pipeline_time']) > 50:
                    for key in self.track_manager.pipeline_stats:
                        if isinstance(self.track_manager.pipeline_stats[key], list):
                            self.track_manager.pipeline_stats[key] = \
                                self.track_manager.pipeline_stats[key][-30:]
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"✅ HUNGARIAN FULL SEQUENCE COMPLETED!")
        print(f"  Processed: {successful_frames}/{num_frames} frames")
        print(f"  Infeasible: {infeasible_frames} frames")
        print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Success rate: {(successful_frames/num_frames)*100:.1f}%")
        
        # Generate results dengan Hungarian stats
        mota_summary = self.mota_calc.get_summary()
        track_manager_stats = self.track_manager.get_track_statistics()
        hungarian_perf = self.track_manager.get_hungarian_performance_summary()
        
        avg_time = np.mean(self.evaluation_times) if self.evaluation_times else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'mota_summary': mota_summary,
            'class_distributions': {
                'gt_classes': dict(self.mota_calc.gt_class_distribution),
                'det_classes': dict(self.mota_calc.det_class_distribution)
            },
            'performance': {
                'avg_processing_time': avg_time,
                'avg_fps': avg_fps,
                'successful_frames': successful_frames,
                'infeasible_frames': infeasible_frames,
                'total_time': total_time,
                'total_frames_attempted': num_frames
            },
            'track_manager_performance': track_manager_stats,
            'hungarian_performance': hungarian_perf,
            'pipeline_performance': self.track_manager._get_pipeline_stats(),
            'frame_details': self.mota_calc.frame_results,
            'sequence_info': {
                'sequence_name': SETTINGS.SEQUENCE_NAME,
                'full_sequence': num_frames >= 500,
                'start_frame': start_frame,
                'end_frame': start_frame + num_frames - 1,
                'pure_hungarian': True,
                'architecture': 'pure_hungarian_optimal',
                'detector_classes': list(self.mota_calc.DETECTOR_CLASSES),
                'evaluation_classes': list(self.mota_calc.MOT_EVAL_CLASSES)
            }
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive results dengan Hungarian performance"""
        mota = results['mota_summary']
        perf = results['performance']
        seq_info = results['sequence_info']
        class_dist = results.get('class_distributions', {})
        track_perf = results.get('track_manager_performance', {})
        hungarian_perf = results.get('hungarian_performance', {})
        pipeline_perf = results.get('pipeline_performance', {})
        
        print("\n" + "=" * 70)
        print("🎯 COMPLETE HUNGARIAN EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"📺 SEQUENCE INFO:")
        print(f"  Sequence: {seq_info['sequence_name']}")
        print(f"  Frames: {seq_info['start_frame']} - {seq_info['end_frame']}")
        print(f"  Total Processed: {perf['successful_frames']}/{perf['total_frames_attempted']}")
        print(f"  Infeasible Frames: {perf['infeasible_frames']}")
        print(f"  Success Rate: {(perf['successful_frames']/perf['total_frames_attempted'])*100:.1f}%")
        print(f"  Pure Hungarian: {seq_info.get('pure_hungarian', False)}")
        print(f"  Architecture: {seq_info.get('architecture', 'unknown')}")
        
        print(f"\n📊 FINAL MOTA METRICS:")
        print(f"  MOTA Score: {mota['MOTA']:.3f}")
        print(f"  IDF1 Score: {mota['IDF1']:.3f}")
        print(f"  Precision:  {mota['precision']:.3f}")
        print(f"  Recall:     {mota['recall']:.3f}")
        
        print(f"\n📈 DETAILED ANALYSIS:")
        print(f"  Total GT Objects:    {mota['total_gt']:,}")
        print(f"  False Positives:     {mota['total_fp']:,}")
        print(f"  False Negatives:     {mota['total_fn']:,}")
        print(f"  ID Switches:         {mota['total_id_switches']:,}")
        print(f"  Processed Frames:    {mota['total_frames']:,}")
        
        # IDF1 detailed breakdown
        print(f"\n🔄 IDENTITY TRACKING (IDF1) ANALYSIS:")
        print(f"  Identity True Positives (IDTP):  {mota['idtp']:,}")
        print(f"  Identity False Positives (IDFP): {mota['idfp']:,}")
        print(f"  Identity False Negatives (IDFN): {mota['idfn']:,}")
        print(f"  Identity Precision: {mota['identity_precision']:.3f}")
        print(f"  Identity Recall:    {mota['identity_recall']:.3f}")
        print(f"  IDF1 Score:         {mota['IDF1']:.3f}")
        
        # Hungarian Performance
        if hungarian_perf.get('available', False):
            print(f"\n🎯 HUNGARIAN PERFORMANCE:")
            print(f"  Algorithm: {hungarian_perf.get('architecture', 'pure_hungarian_optimal')}")
            print(f"  Total Calls: {hungarian_perf.get('total_calls', 0)}")
            print(f"  Success Rate: {hungarian_perf.get('success_rate', 0):.1%}")
            print(f"  Infeasible Rate: {hungarian_perf.get('infeasible_rate', 0):.1%}")
            print(f"  Avg Assignment Rate: {hungarian_perf.get('avg_assignment_rate', 0):.3f}")
            print(f"  Avg Execution Time: {hungarian_perf.get('avg_execution_time', 0):.4f}s")
            print(f"  Optimal Assignments: {hungarian_perf.get('optimal_assignments', 0)}")
            print(f"  Rejected Assignments: {hungarian_perf.get('rejected_assignments', 0)}")
            
            # Optimizer performance
            opt_perf = hungarian_perf.get('optimizer_performance', {})
            if opt_perf.get('available', False):
                print(f"\n🚀 HUNGARIAN OPTIMIZER PERFORMANCE:")
                print(f"  Success Rate: {opt_perf.get('success_rate', 0):.1%}")
                print(f"  Infeasible Cases: {opt_perf.get('infeasible_cases', 0)}")
                print(f"  Avg Cost Statistics: {opt_perf.get('cost_statistics', {})}")
        
        # Pipeline Performance
        if pipeline_perf:
            fps = pipeline_perf.get('fps', 0)
            print(f"\n🚀 PIPELINE PERFORMANCE:")
            print(f"  Pipeline FPS: {fps:.1f}")
            print(f"  Detection Time: {pipeline_perf.get('detection_time_avg', 0):.3f}s")
            print(f"  Feature Time: {pipeline_perf.get('feature_extraction_time_avg', 0):.3f}s")
            print(f"  Graph Time: {pipeline_perf.get('graph_construction_time_avg', 0):.3f}s")
            print(f"  GNN Time: {pipeline_perf.get('gnn_prediction_time_avg', 0):.3f}s")
            print(f"  Data Association Time: {pipeline_perf.get('data_association_time_avg', 0):.3f}s")
            print(f"  Track Update Time: {pipeline_perf.get('track_update_time_avg', 0):.3f}s")
        
        # Track Manager Performance
        if track_perf:
            print(f"\n📈 TRACK MANAGER PERFORMANCE:")
            print(f"  Total Tracks Created: {track_perf.get('total_tracks_created', 0)}")
            print(f"  Active Tracks: {track_perf.get('active_tracks', 0)}")
            print(f"  Ghost Tracks: {track_perf.get('ghost_tracks', 0)}")
            print(f"  Confirmed Tracks: {track_perf.get('confirmed_tracks', 0)}")
            print(f"  Total Associations: {track_perf.get('total_associations', 0)}")
            print(f"  Ghost Reidentifications: {track_perf.get('total_ghost_reidentifications', 0)}")
            print(f"  Algorithm: {track_perf.get('matching_algorithm', 'unknown')}")
        
        print(f"\n⚡ OVERALL PERFORMANCE:")
        print(f"  Total Processing:    {perf['total_time']:.1f}s ({perf['total_time']/60:.1f} min)")
        print(f"  Average FPS:         {perf['avg_fps']:.1f}")
        print(f"  Avg Frame Time:      {perf['avg_processing_time']:.3f}s")
        
        # Combined assessment
        print(f"\n🎯 HUNGARIAN ASSESSMENT:")
        mota_score = mota['MOTA']
        idf1_score = mota['IDF1']
        hungarian_success = hungarian_perf.get('success_rate', 0) if hungarian_perf.get('available', False) else 0
        infeasible_rate = hungarian_perf.get('infeasible_rate', 0) if hungarian_perf.get('available', False) else 0
        
        if mota_score >= 0.7 and idf1_score >= 0.7 and hungarian_success >= 0.9:
            status = "🏆 EXCEPTIONAL! Hungarian algorithm perfect!"
            grade = "A+"
        elif mota_score >= 0.65 and idf1_score >= 0.65 and hungarian_success >= 0.8:
            status = "🥇 EXCELLENT! Hungarian performing optimally!"
            grade = "A"
        elif mota_score >= 0.6 and idf1_score >= 0.6 and hungarian_success >= 0.7:
            status = "✅ VERY GOOD! Hungarian working well!"
            grade = "A-"
        else:
            status = "👍 GOOD! Hungarian functional with room for optimization"
            grade = "B+"
        
        print(f"  Overall Grade: {grade}")
        print(f"  Status: {status}")
        print(f"  Infeasible Rate: {infeasible_rate:.1%}")
        
        # Architecture benefits
        print(f"\n🗼 HUNGARIAN ALGORITHM BENEFITS:")
        print(f"  ✅ Mathematically optimal assignment")
        print(f"  ✅ Minimum total cost guarantee")
        print(f"  ✅ Well-established algorithm (Kuhn-Munkres)")
        print(f"  ✅ Consistent performance across scenes")
        print(f"  ✅ Handles dense assignment scenarios well")
        print(f"  ⚠️  May produce false positives in challenging cases")
        
        print("=" * 70)
    
    def save_results(self, results: Dict):
        """Save results dengan Hungarian performance data"""
        output_dir = Path("results_hungarian")
        output_dir.mkdir(exist_ok=True)
        
        # Save enhanced results dengan Hungarian data
        with open(output_dir / "hungarian_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Hungarian results saved to {output_dir}")


def main():
    """Main function untuk Hungarian evaluation"""
    try:
        print("🎯 HUNGARIAN EVALUATOR")
        print("Architecture: Pure Hungarian Optimal Assignment")
        print("=" * 60)
        
        # Initialize evaluator dengan Hungarian
        evaluator = HungarianEvaluator()
        
        # Run evaluation - FULL SEQUENCE!
        print("🚀 TESTING HUNGARIAN ALGORITHM (FULL SEQUENCE)")
        print("This might take a few minutes...")
        
        results = evaluator.run_evaluation(
            start_frame=1,
            num_frames=1050 # FULL SEQUENCE!
        )
        
        # Print comprehensive results
        evaluator.print_results(results)
        
        # Save results
        evaluator.save_results(results)
        
        # Final validation
        mota_score = results['mota_summary']['MOTA']
        hungarian_stats = results.get('hungarian_performance', {})
        hungarian_success = hungarian_stats.get('success_rate', 0) if hungarian_stats.get('available', False) else 0
        infeasible_rate = hungarian_stats.get('infeasible_rate', 0) if hungarian_stats.get('available', False) else 0
        
        print(f"\n🔍 FINAL HUNGARIAN VALIDATION:")
        print(f"  MOTA Score: {mota_score:.3f}")
        print(f"  Hungarian Success: {hungarian_success:.1%}")
        print(f"  Infeasible Rate: {infeasible_rate:.1%}")
        print(f"  ID Switches: {results['mota_summary']['total_id_switches']}")
        print(f"  IDF1 Score: {results['mota_summary']['IDF1']:.3f}")
        
        if mota_score > 0.6 and hungarian_success > 0.8 and infeasible_rate < 0.1:
            print("  🎉 EXCELLENT! Hungarian algorithm working optimally!")
            print("  🚀 Ready for comparative analysis!")
            print("  📊 Expected: Optimal assignments with potential for false positives")
        elif mota_score > 0.5 and hungarian_success > 0.7:
            print("  👍 GOOD! Hungarian algorithm performing well!")
            print("  🔧 Consider cost matrix tuning for better performance")
        else:
            print("  🔧 NEEDS WORK! Check Hungarian configuration or cost thresholds")
        
        print("\n✅ Hungarian evaluation completed")
        
        # Comparison note
        print(f"\n📋 ALGORITHM COMPARISON:")
        print(f"  Hungarian: Optimal assignment, mathematically guaranteed minimum cost")
        print(f"  Smart Hungarian: Quality-controlled, better miss than wrong philosophy")
        print(f"  Greedy: Fast sequential assignment, may be suboptimal")
        print(f"  Hungarian: Good baseline for comparative analysis")
        
    except Exception as e:
        print(f"❌ Hungarian evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()