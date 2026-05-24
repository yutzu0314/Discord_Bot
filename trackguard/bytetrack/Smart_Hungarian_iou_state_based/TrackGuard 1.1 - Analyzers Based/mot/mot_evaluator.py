"""
TrackGuard MOT Evaluator - TrackManager Two-Stage SHA (State-Based)
===================================================================

State-Based Two-Stage dengan Smart Hungarian TrackGuard:
- Stage 1: Active tracks (state='active') → Smart Hungarian matching
- Stage 2: Ghost tracks (state='ghost') → Ghost recovery matching

TUJUAN:
1. Filter GT untuk kelas 1,2,7 (person categories)  
2. Filter deteksi untuk person saja
3. Test State-Based Two-Stage + Smart Hungarian TrackGuard
4. Hitung MOTA, IDF1, FP, FN, ID-Switch yang benar
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path
import json
from collections import defaultdict

# Import komponen TrackGuard
from src.models.yolo_detector import YOLODetector
from src.core.track_manager_two_stage_sha import TrackManagerTwoStageSHA
from src.core.bbox_handler import BBoxHandler
from src.core.confidence_handler import ConfidenceHandler
from src.analyzers.color_analyzer import ColorAnalyzer
from src.analyzers.shape_analyzer import ShapeAnalyzer
from src.config.trackguard_config import TrackGuardConfig
from src.utils.mot17_loader import MOT17Loader

# KONFIGURASI EVALUATOR - EDIT BAGIAN INI
CONFIG = {
    # Path konfigurasi
    'detector_path': 'best.pt',          # Path ke model YOLO
    'dataset_root': 'MOT17/',            # Path ke dataset MOT17
    'sequence': 'MOT17-13-SDP',          # Sequence yang akan dievaluasi
    'config_path': 'config/optimal_config.json', # Path ke TrackGuard config (untuk compatibility)
    
    # Evaluasi parameters
    'eval_classes': [1, 2, 7],          # Kelas yang dievaluasi (person categories)
    'start_frame': 1,                    # Frame mulai
    'num_frames': 750,                   # Jumlah frame (750 untuk full sequence)
    'iou_threshold': 0.5,                # IoU threshold untuk matching
    
    # YOLO parameters
    'conf_threshold': 0.25,              # Confidence threshold YOLO
    'iou_nms': 0.4,                      # IoU NMS threshold
    'device': 'cuda',                    # Device untuk inference
    
    # TrackManagerTwoStageSHA parameters
    'max_age': 30,                       # Maximum frames to keep unmatched tracks
    'min_hits': 3,                       # Minimum hits for track confirmation
    'fps': 30,                           # Frames per second
    'enable_ghost': True,                # Enable ghost node mechanism
}

class SimpleMOTACalculator:
    """
    Simple MOTA Calculator - fokus pada akurasi hasil
    Sama persis dengan evaluator lainnya
    """
    
    def __init__(self):
        # MOT Challenge classes yang dievaluasi
        self.MOT_EVAL_CLASSES = {1, 2, 7}  # person, person_on_vehicle, static_person
        
        # TrackGuard detector classes (YOLO output classes)
        self.DETECTOR_CLASSES = {0, 1, 2}  # YOLO classes: 0=person, 1=person_on_vehicle, 2=static_person
        
        # Mapping dari YOLO class ke MOT eval class
        self.CLASS_MAPPING = {
            0: 1,  # YOLO person (0) → MOT person (1)
            1: 2,  # YOLO person_on_vehicle (1) → MOT person_on_vehicle (2)  
            2: 7   # YOLO static_person (2) → MOT static_person (7)
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
        
        # Class distribution tracking
        self.gt_class_distribution = defaultdict(int)
        self.det_class_distribution = defaultdict(int)
        
        algorithm_name = "State-Based Two-Stage SHA"
        print(f"✓ MOTA Calculator reset (TrackGuard {algorithm_name} mode + IDF1)")
        print(f"  Detector classes: {self.DETECTOR_CLASSES}")
        print(f"  Evaluation classes: {self.MOT_EVAL_CLASSES}")
        print(f"  Class mapping: {self.CLASS_MAPPING}")
    
    def filter_gt_annotations(self, gt_annotations: List[Dict]) -> List[Dict]:
        """Filter GT untuk kelas MOT Challenge (1, 2, 7)"""
        filtered = []
        
        for gt in gt_annotations:
            class_id = gt.get('class_id', 1)
            
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
        """Filter deteksi TrackGuard"""
        filtered = []
        
        for det in detections:
            det_class = det.get('class_id', 1)
            
            try:
                det_class = int(det_class)
            except:
                det_class = 1  # default person
            
            # Track distribution
            self.det_class_distribution[det_class] += 1
            
            # Filter berdasarkan classes
            if det_class in self.DETECTOR_CLASSES:
                confidence = float(det.get('confidence', 0.8))
                if confidence >= 0.3:  # Minimum confidence
                    # Map detector class ke evaluation class
                    mapped_class = self.CLASS_MAPPING.get(det_class, det_class)
                    det_copy = det.copy()
                    det_copy['mapped_class'] = mapped_class
                    det_copy['original_class'] = det_class
                    filtered.append(det_copy)
        
        return filtered
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
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
        """Match detections to ground truth menggunakan IoU"""
        if not gt_list or not det_list:
            return [], len(det_list), len(gt_list)
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(det_list), len(gt_list)))
        
        for i, det in enumerate(det_list):
            det_bbox = det.get('bbox', [])
            if len(det_bbox) != 4:
                continue
                
            for j, gt in enumerate(gt_list):
                gt_bbox = gt.get('bbox', [])
                if len(gt_bbox) != 4:
                    continue
                
                iou = self.compute_iou(det_bbox, gt_bbox)
                iou_matrix[i, j] = iou
        
        # Simple greedy matching
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
        """Count ID switches berdasarkan matches"""
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
                    print(f"  ID Switch: Track {track_id}: GT {prev_gt_id} → {gt_id}")
        
        self.prev_matches = current_matches
        return switches
    
    def update_idf1_tracking(self, matches: List[Dict]):
        """Update IDF1 tracking based on current frame matches"""
        for match in matches:
            track_id = match['track_id']
            gt_id = match['gt_id']
            
            # Track which GT IDs each track has been assigned to
            self.track_gt_assignments[track_id].append(gt_id)
            
            # Track which track IDs each GT has been assigned to  
            self.gt_track_assignments[gt_id].append(track_id)
    
    def compute_idf1_components(self):
        """Compute IDF1 components: IDTP, IDFP, IDFN"""
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
    
    def update_frame(self, gt_annotations: List[Dict], tracking_results: List[Dict], 
                    frame_id: int) -> Dict:
        """Update MOTA calculation untuk satu frame"""
        # Filter data sesuai MOT Challenge
        filtered_gt = self.filter_gt_annotations(gt_annotations)
        
        # Convert tracking results ke format deteksi
        detections = []
        for track in tracking_results:
            det = {
                'track_id': track['track_id'],
                'bbox': track['bbox'],
                'confidence': track.get('confidence', 0.8),
                'class_id': track.get('class_id', 1)
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
        if frame_id % 50 == 1:  # Print every 50 frames
            mota = self.get_mota()
            idf1 = self.get_idf1()
            print(f"Frame {frame_id}: GT={num_gt}, Det={len(filtered_detections)}, "
                  f"Match={len(matches)}, FP={fp}, FN={fn}, IDSW={id_switches}, "
                  f"MOTA={mota:.3f}, IDF1={idf1:.3f}")
                  
            # Progress indicator
            if frame_id > 1:
                progress = ((frame_id - 1) / 750) * 100
                print(f"  Progress: {progress:.1f}% - MOTA: {mota:.3f}, IDF1: {idf1:.3f}")
        
        return frame_result
    
    def get_idf1(self) -> float:
        """Calculate IDF1 score"""
        idtp, idfp, idfn = self.compute_idf1_components()
        
        if idtp == 0:
            return 0.0
        
        idf1 = (2 * idtp) / (2 * idtp + idfp + idfn)
        return max(0.0, min(1.0, idf1))
    
    def get_mota(self) -> float:
        """Calculate MOTA score"""
        if self.total_gt == 0:
            return 0.0
        
        # MOTA = 1 - (FN + FP + IDSW) / GT
        mota = 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / self.total_gt
        return max(0.0, mota)
    
    def get_summary(self) -> Dict:
        """Get evaluation summary WITH IDF1"""
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


class TrackGuardMOTEvaluatorSHA:
    """
    TrackGuard MOT Evaluator - State-Based Two-Stage dengan Smart Hungarian TrackGuard
    """
    
    def __init__(self):
        algorithm_name = "State-Based Two-Stage SHA"
        print(f"=== TRACKGUARD MOT EVALUATOR ({algorithm_name.upper()}) ===")
        print("Fokus: MOTA evaluation dengan state-based two-stage + Smart Hungarian TrackGuard")
        
        # Initialize TrackGuard komponen dengan config file (untuk compatibility)
        # Note: TrackManagerTwoStageSHA tidak menggunakan config, tapi kita tetap load untuk compatibility
        self.config = TrackGuardConfig(CONFIG['config_path'])
        
        # Initialize YOLO detector
        self.detector = YOLODetector(
            CONFIG['detector_path'], 
            conf_threshold=CONFIG['conf_threshold'],
            iou_threshold=CONFIG['iou_nms'],
            device=CONFIG['device']
        )
        
        # Initialize TrackGuard components
        self.bbox_handler = BBoxHandler()
        self.confidence_handler = ConfidenceHandler()
        self.color_analyzer = ColorAnalyzer()
        self.shape_analyzer = ShapeAnalyzer()
        
        # Initialize TrackManagerTwoStageSHA (State-Based Two-Stage)
        self.track_manager = TrackManagerTwoStageSHA(
            color_analyzer=self.color_analyzer,
            shape_analyzer=self.shape_analyzer,
            max_age=CONFIG['max_age'],
            min_hits=CONFIG['min_hits'],
            fps=CONFIG['fps'],
            enable_ghost=CONFIG['enable_ghost']
        )
        
        # Initialize dataset loader
        self.data_loader = MOT17Loader(CONFIG['dataset_root'], CONFIG['sequence'])
        
        # MOTA calculator
        self.mota_calc = SimpleMOTACalculator()
        
        # Performance tracking
        self.frame_times = []
        
        # Algorithm info
        algorithm_features = ["State-Based Two-Stage", "Smart Hungarian TrackGuard", "Ghost Recovery"]
        
        print(f"✓ TrackGuard MOT evaluator initialized")
        print(f"  Algorithm: {' + '.join(algorithm_features)}")
        print(f"  Dataset: {CONFIG['sequence']}")
        print(f"  Eval classes: {CONFIG['eval_classes']}")
        print(f"  Total frames: {len(self.data_loader)}")
        print(f"  Max age: {CONFIG['max_age']}, Min hits: {CONFIG['min_hits']}")
        print(f"  Ghost nodes: {CONFIG['enable_ghost']}")
        print(f"  Note: Tuning parameters hardcoded in smart_hungarian_trackguard.py")
    
    def process_frame(self, frame_id: int) -> Optional[Dict]:
        """Process single frame untuk tracking dan evaluation"""
        # Load frame dan GT
        image = self.data_loader.get_frame_image(frame_id)
        gt_annotations = self.data_loader.get_frame_annotations(frame_id)
        
        if image is None:
            return None
        
        # 1. YOLO Detection
        yolo_detections = self.detector.detect(image)
        
        # 2. TrackGuard Processing
        # Convert YOLO detections ke format TrackGuard
        trackguard_detections = []
        for det in yolo_detections:
            trackguard_det = {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_id': det['class_id']
            }
            trackguard_detections.append(trackguard_det)
        
        # 3. Update tracks dengan TrackManagerTwoStageSHA
        active_tracks = self.track_manager.update(image, trackguard_detections)
        
        # 4. Enhance tracks dengan confidence handler
        enhanced_tracks = []
        for track in active_tracks:
            # Get analysis region
            analysis_bbox = self.bbox_handler.get_analysis_region(track['bbox'])
            
            # Analyze region  
            color_score = self.color_analyzer.analyze(image, analysis_bbox, track['track_id'])
            shape_score = self.shape_analyzer.analyze(image, analysis_bbox, track['track_id'])
            
            # Update confidence
            enhanced_conf = self.confidence_handler.update_confidence(
                track['track_id'],
                track['confidence'],
                color_score,
                shape_score, 
                track['history']
            )
            
            # Create enhanced track
            enhanced_track = {
                'track_id': track['track_id'],
                'bbox': track['bbox'],
                'confidence': enhanced_conf,
                'class_id': track['class_id']
            }
            enhanced_tracks.append(enhanced_track)
        
        return {
            'frame_id': frame_id,
            'image': image,
            'yolo_detections': yolo_detections,
            'active_tracks': enhanced_tracks,
            'gt_annotations': gt_annotations or []
        }
    
    def run_evaluation(self, start_frame: int = None, num_frames: int = None) -> Dict:
        """Run TrackGuard MOT evaluation dengan State-Based Two-Stage SHA"""
        start_frame = start_frame or CONFIG['start_frame']
        num_frames = num_frames or CONFIG['num_frames']
        
        algorithm_name = "State-Based Two-Stage SHA"
        print(f"\n🚀 Starting TrackGuard {algorithm_name} MOTA Evaluation")
        print(f"Frames: {start_frame} to {start_frame + num_frames - 1}")
        print(f"Expected time: ~{(num_frames * 0.15 / 60):.1f} minutes")
        print("=" * 60)
        
        # Reset
        self.mota_calc.reset()
        self.track_manager = TrackManagerTwoStageSHA(
            color_analyzer=self.color_analyzer,
            shape_analyzer=self.shape_analyzer,
            max_age=CONFIG['max_age'],
            min_hits=CONFIG['min_hits'],
            fps=CONFIG['fps'],
            enable_ghost=CONFIG['enable_ghost']
        )
        self.frame_times = []
        
        successful_frames = 0
        start_time = time.time()
        
        for i in range(num_frames):
            frame_id = start_frame + i
            
            frame_start_time = time.time()
            
            # Process frame
            frame_result = self.process_frame(frame_id)
            if frame_result is None:
                print(f"⚠️ Frame {frame_id} failed to load")
                continue
            
            processing_time = time.time() - frame_start_time
            self.frame_times.append(processing_time)
            
            # Update MOTA calculation
            self.mota_calc.update_frame(
                frame_result['gt_annotations'],
                frame_result['active_tracks'],
                frame_id
            )
            
            successful_frames += 1
            
            # Memory management untuk full sequence
            if successful_frames % 100 == 0:
                # Cleanup stale tracks
                active_track_ids = [t['track_id'] for t in frame_result['active_tracks']]
                self.color_analyzer.cleanup_stale_tracks(active_track_ids)
                self.shape_analyzer.cleanup_stale_tracks(active_track_ids)
                self.confidence_handler.cleanup_stale_tracks(active_track_ids)
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"✓ TRACKGUARD {algorithm_name.upper()} EVALUATION COMPLETED!")
        print(f"  Processed: {successful_frames}/{num_frames} frames")
        print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Success rate: {(successful_frames/num_frames)*100:.1f}%")
        
        # Generate results
        mota_summary = self.mota_calc.get_summary()
        
        avg_time = np.mean(self.frame_times) if self.frame_times else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Enhanced summary
        results = {
            'mota_summary': mota_summary,
            'class_distributions': {
                'gt_classes': dict(self.mota_calc.gt_class_distribution),
                'det_classes': dict(self.mota_calc.det_class_distribution)
            },
            'performance': {
                'avg_processing_time': avg_time,
                'avg_fps': avg_fps,
                'successful_frames': successful_frames,
                'total_time': total_time,
                'total_frames_attempted': num_frames
            },
            'frame_details': self.mota_calc.frame_results,
            'sequence_info': {
                'sequence_name': CONFIG['sequence'],
                'start_frame': start_frame,
                'end_frame': start_frame + num_frames - 1,
                'trackguard_config': self.config.get_summary(),
                'eval_classes': CONFIG['eval_classes'],
                'algorithm_config': {
                    'algorithm_type': 'state_based_two_stage_sha',
                    'matcher': 'smart_hungarian_trackguard',
                    'max_age': CONFIG['max_age'],
                    'min_hits': CONFIG['min_hits'],
                    'enable_ghost': CONFIG['enable_ghost'],
                    'note': 'Tuning parameters hardcoded in smart_hungarian_trackguard.py'
                }
            }
        }
        
        # Print dan save results
        self.print_results(results)
        self.save_results(results)
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results dengan algorithm info"""
        mota = results['mota_summary']
        perf = results['performance']
        seq_info = results['sequence_info']
        class_dist = results.get('class_distributions', {})
        algo_config = seq_info.get('algorithm_config', {})
        
        algorithm_name = "State-Based Two-Stage SHA"
        
        print("\n" + "=" * 70)
        print(f"🏁 TRACKGUARD {algorithm_name.upper()} MOT EVALUATION RESULTS")  
        print("=" * 70)
        
        print(f"📋 SEQUENCE INFO:")
        print(f"  Sequence: {seq_info['sequence_name']}")
        print(f"  Frames: {seq_info['start_frame']} - {seq_info['end_frame']}")
        print(f"  Total Processed: {perf['successful_frames']}/{perf['total_frames_attempted']}")
        print(f"  Success Rate: {(perf['successful_frames']/perf['total_frames_attempted'])*100:.1f}%")
        
        # Algorithm configuration info
        print(f"\n⚙️ ALGORITHM CONFIGURATION:")
        print(f"  Algorithm: {algorithm_name}")
        print(f"  Matcher: Smart Hungarian TrackGuard")
        print(f"  Stage 1: Active tracks (state='active') → Smart Hungarian matching")
        print(f"  Stage 2: Ghost tracks (state='ghost') → Ghost recovery matching")
        print(f"  Max Age: {algo_config.get('max_age', 'N/A')}")
        print(f"  Min Hits: {algo_config.get('min_hits', 'N/A')}")
        print(f"  Ghost Nodes: {algo_config.get('enable_ghost', 'N/A')}")
        print(f"  Note: {algo_config.get('note', '')}")
        
        # Class distribution info
        print(f"\n📊 CLASS DISTRIBUTIONS:")
        gt_classes = class_dist.get('gt_classes', {})
        det_classes = class_dist.get('det_classes', {})
        print(f"  GT Classes Found: {gt_classes}")
        print(f"  Detection Classes: {det_classes}")
        print(f"  Evaluation Classes: {seq_info.get('eval_classes', [1,2,7])}")
        
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
        print(f"\n🔍 IDENTITY TRACKING (IDF1) ANALYSIS:")
        print(f"  Identity True Positives (IDTP):  {mota['idtp']:,}")
        print(f"  Identity False Positives (IDFP): {mota['idfp']:,}")
        print(f"  Identity False Negatives (IDFN): {mota['idfn']:,}")
        print(f"  Identity Precision: {mota['identity_precision']:.3f}")
        print(f"  Identity Recall:    {mota['identity_recall']:.3f}")
        print(f"  IDF1 Score:         {mota['IDF1']:.3f}")
        
        # Per-frame averages
        print(f"\n📊 PER-FRAME AVERAGES:")
        print(f"  GT per frame:        {mota['avg_gt_per_frame']:.1f}")
        print(f"  FP per frame:        {mota['avg_fp_per_frame']:.1f}")
        print(f"  FN per frame:        {mota['avg_fn_per_frame']:.1f}")
        print(f"  ID switches/100f:    {(mota['total_id_switches']/mota['total_frames'])*100:.1f}")
        
        print(f"\n⚡ PERFORMANCE:")
        print(f"  Total Processing:    {perf['total_time']:.1f}s ({perf['total_time']/60:.1f} min)")
        print(f"  Average FPS:         {perf['avg_fps']:.1f}")
        print(f"  Avg Frame Time:      {perf['avg_processing_time']:.3f}s")
        
        print(f"\n🎯 TRACKGUARD ASSESSMENT:")
        mota_score = mota['MOTA']
        idf1_score = mota['IDF1']
        id_switches = mota['total_id_switches']
        
        # Assessment dengan IDF1
        if mota_score >= 0.7 and idf1_score >= 0.7 and id_switches <= 50:
            status = "🏆 EXCEPTIONAL! TrackGuard perfect performance!"
            grade = "A+"
        elif mota_score >= 0.65 and idf1_score >= 0.65 and id_switches <= 100:
            status = "🎉 EXCELLENT! Strong MOTA + IDF1 performance!"
            grade = "A"
        elif mota_score >= 0.6 and idf1_score >= 0.6 and id_switches <= 150:
            status = "✅ VERY GOOD! Solid MOTA + IDF1 scores!"
            grade = "A-"
        elif mota_score >= 0.55 and idf1_score >= 0.5:
            status = "👍 GOOD! MOTA strong, IDF1 decent!"
            grade = "B+"
        else:
            status = "⚠️ NEEDS ANALYSIS - Check MOTA vs IDF1 balance"
            grade = "B"
        
        print(f"  Overall Grade: {grade}")
        print(f"  Status: {status}")
        
        # TrackGuard specific analysis
        print(f"\n🔧 TRACKGUARD {algorithm_name.upper()} ANALYSIS:")
        config_summary = seq_info.get('trackguard_config', {})
        print(f"  Config Parameters: {config_summary.get('total_parameters', 'N/A')}")
        print(f"  Using Optimal Config: {config_summary.get('using_optimal_defaults', True)}")
        print(f"  Algorithm: State-Based Two-Stage")
        print(f"  Matcher: Smart Hungarian TrackGuard")
        print(f"  Cost Matrix: Multi-Modal (IoU + Color + Shape)")
        print(f"  Weighting: Scene-Adaptive (in smart_hungarian_trackguard.py)")
        
        print("=" * 70)
    
    def save_results(self, results: Dict):
        """Save results to file dengan algorithm info"""
        algorithm_name = "statebased_twostage_sha"
        
        output_dir = Path(f"results_trackguard_{algorithm_name}")
        output_dir.mkdir(exist_ok=True)
        
        # Save results JSON
        with open(output_dir / f"trackguard_{algorithm_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary text
        with open(output_dir / f"trackguard_{algorithm_name}_summary.txt", 'w') as f:
            mota = results['mota_summary']
            seq_info = results.get('sequence_info', {})
            perf = results['performance']
            algo_config = seq_info.get('algorithm_config', {})
            
            f.write(f"TRACKGUARD STATE-BASED TWO-STAGE SHA MOT EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Algorithm: State-Based Two-Stage + Smart Hungarian TrackGuard\n")
            f.write(f"Sequence: {seq_info.get('sequence_name', 'Unknown')}\n")
            f.write(f"Frames: {seq_info.get('start_frame', 1)} - {seq_info.get('end_frame', 'N/A')}\n")
            f.write(f"Success Rate: {(perf['successful_frames']/perf['total_frames_attempted'])*100:.1f}%\n\n")
            
            # Algorithm config
            f.write(f"Algorithm Type: {algo_config.get('algorithm_type', 'N/A')}\n")
            f.write(f"Matcher: {algo_config.get('matcher', 'N/A')}\n")
            f.write(f"Max Age: {algo_config.get('max_age', 'N/A')}\n")
            f.write(f"Min Hits: {algo_config.get('min_hits', 'N/A')}\n")
            f.write(f"Ghost Nodes: {algo_config.get('enable_ghost', 'N/A')}\n")
            f.write(f"Note: {algo_config.get('note', '')}\n")
            f.write(f"Cost Matrix: Multi-Modal (IoU + Color + Shape)\n")
            f.write(f"Weighting: Scene-Adaptive (in smart_hungarian_trackguard.py)\n")
            f.write("\n")
            
            f.write(f"MOTA Score: {mota['MOTA']:.3f}\n")
            f.write(f"IDF1 Score: {mota['IDF1']:.3f}\n")
            f.write(f"Precision:  {mota['precision']:.3f}\n")
            f.write(f"Recall:     {mota['recall']:.3f}\n\n")
            f.write(f"Total GT:   {mota['total_gt']:,}\n")
            f.write(f"FP:         {mota['total_fp']:,}\n")
            f.write(f"FN:         {mota['total_fn']:,}\n")
            f.write(f"ID Switch:  {mota['total_id_switches']:,}\n")
            f.write(f"Frames:     {mota['total_frames']:,}\n\n")
            f.write(f"Processing Time: {perf['total_time']:.1f}s ({perf['total_time']/60:.1f} min)\n")
            f.write(f"Average FPS: {perf['avg_fps']:.1f}\n")
        
        print(f"✓ TrackGuard {algorithm_name} results saved to {output_dir}")


def main():
    """Main function untuk TrackGuard State-Based Two-Stage SHA MOT evaluation"""
    try:
        algorithm_name = "State-Based Two-Stage SHA"
        print(f"🚀 TRACKGUARD {algorithm_name.upper()} MOT EVALUATOR")
        print("Tujuan: MOTA evaluation dengan state-based two-stage + Smart Hungarian TrackGuard")
        print("=" * 60)
        
        # Display algorithm configuration
        print("⚙️ ALGORITHM CONFIGURATION:")
        print(f"  Algorithm: {algorithm_name}")
        print(f"  Matcher: Smart Hungarian TrackGuard")
        print(f"  Stage 1: Active tracks (state='active') → Smart Hungarian matching")
        print(f"  Stage 2: Ghost tracks (state='ghost') → Ghost recovery matching")
        print(f"  Cost Matrix: Multi-Modal (IoU + Color + Shape)")
        print(f"  Weighting: Scene-Adaptive (tuning in smart_hungarian_trackguard.py)")
        print(f"  Max Age: {CONFIG['max_age']}")
        print(f"  Min Hits: {CONFIG['min_hits']}")
        print(f"  Ghost Nodes: {CONFIG['enable_ghost']}")
        print("=" * 60)
        
        # Validate config
        if CONFIG['detector_path'] == 'path/to/best.pt':
            print("❌ Error: Please update CONFIG['detector_path'] with actual path to model")
            return
            
        if CONFIG['dataset_root'] == 'path/to/MOT17':
            print("❌ Error: Please update CONFIG['dataset_root'] with actual path to MOT17")
            return
        
        # Initialize evaluator
        evaluator = TrackGuardMOTEvaluatorSHA()
        
        # Run evaluation
        print(f"🚀 Running TrackGuard {algorithm_name} evaluation...")
        
        results = evaluator.run_evaluation()
        
        # Final validation
        mota_score = results['mota_summary']['MOTA']
        idf1_score = results['mota_summary']['IDF1']
        
        print(f"\n🔍 FINAL VALIDATION:")
        print(f"  MOTA Score: {mota_score:.3f}")
        print(f"  IDF1 Score: {idf1_score:.3f}")
        
        if mota_score > 0.01:
            print(f"  ✅ SUCCESS! TrackGuard {algorithm_name} MOTA calculation working")
        else:
            print("  ⚠️  MOTA still low - check data matching")
        
        print(f"\n✓ TrackGuard {algorithm_name} evaluation completed")
        
    except Exception as e:
        print(f"❌ TrackGuard evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

