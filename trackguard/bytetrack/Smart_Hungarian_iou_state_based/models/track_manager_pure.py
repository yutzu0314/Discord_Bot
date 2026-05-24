"""
ByteTrack + Ghost Track TrackManager - State-Based Implementation
=================================================================

TrackManager menggunakan ByteTrack three-stage Hungarian association dengan
state-based ghost track recovery untuk occlusion handling.

FEATURES:
- ByteTrack-style two-stage matching (high-conf + low-conf detections)
- State-based ghost track recovery (NOVELTY!)
- Pure Hungarian algorithm (no quality gates)
- IoU-based cost calculation
- Lightweight dan efficient (target 30+ FPS)

PIPELINE:
YOLOv8 → IoU Calculator → ByteTrack Three-Stage (High + Low + Ghost) → TrackUpdate

NOVELTY:
ByteTrack (stage 1 & 2) + Ghost Track Recovery (stage 3)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import time
import cv2
from collections import defaultdict
import logging

# Import pipeline components
from models.detector import YOLOv8Detector
from models.iou_calculator import IoUCalculator
from models.kalman_filter import KalmanFilter  # ✅ Import dari modul terpisah

# Configure logging with file handler for debug
import os
log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log-debug.txt')

# Setup logger with file handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler untuk debug log
file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler untuk important messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


class Track:
    """Track representation dengan ByteTrack-compatible Kalman Filter"""
    
    def __init__(self, track_id: int, detection: Dict, features: np.ndarray, frame_id: int):
        self.track_id = track_id
        self.state = 'active'
        self.is_activated = False  # Untuk ByteTrack compatibility
        
        # Current state
        self.current_detection = detection
        self.current_features = features
        self.current_frame = frame_id
        
        # Track history
        self.history = [detection]
        self.feature_history = [features] if features is not None else []
        self.frame_history = [frame_id]
        
        # Track statistics
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.last_seen = frame_id
        self.start_frame = frame_id
        
        # Ghost node specific
        self.ghost_start_frame = None
        self.ghost_predictions = []
        self.ghost_confidence = 1.0
        self.initial_confidence = detection.get('confidence', 0.8)
        
        # Kalman Filter (ByteTrack format: 8D state space)
        self.kalman_filter = KalmanFilter()
        self.mean = None  # Kalman mean (8D: x, y, a, h, vx, vy, va, vh)
        self.covariance = None  # Kalman covariance (8x8)
        
        # Initialize Kalman filter dengan detection
        bbox = detection.get('bbox', [0, 0, 1, 1])
        tlwh = self._bbox_to_tlwh(bbox)
        xyah = self._tlwh_to_xyah(tlwh)
        self.mean, self.covariance = self.kalman_filter.initiate(xyah)
        
        # Motion model (legacy, untuk backward compatibility)
        self.velocity = np.array([0.0, 0.0])
        
        # Track quality metrics
        self.avg_confidence = self.initial_confidence
        self.confidence_history = [self.initial_confidence]
        self.stability_score = 1.0
        
        # ByteTrack attributes
        self.bbox = bbox
        self.center = detection.get('center', self._bbox_to_center(bbox))
        self.confidence = self.initial_confidence
        self.score = self.initial_confidence
        self.time_since_update = 0
        self.tracklet_len = 0
    
    @staticmethod
    def _bbox_to_tlwh(bbox):
        """Convert bbox (x1, y1, x2, y2) or (x, y, w, h) to tlwh"""
        if len(bbox) != 4:
            return np.array([0, 0, 1, 1], dtype=float)
        
        # Check if (x1, y1, x2, y2) or (x, y, w, h)
        if bbox[2] > 100 or bbox[3] > 100:  # Likely (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            return np.array([x1, y1, w, h], dtype=float)
        else:  # (x, y, w, h)
            return np.array(bbox, dtype=float)
    
    @staticmethod
    def _bbox_to_center(bbox):
        """Extract center from bbox"""
        if len(bbox) != 4:
            return [0, 0]
        
        if bbox[2] > 100 or bbox[3] > 100:  # (x1, y1, x2, y2)
            return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        else:  # (x, y, w, h)
            return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    
    @staticmethod
    def _tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah format for Kalman Filter"""
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[:2] += ret[2:] / 2  # center = top_left + width/height / 2
        ret[2] /= ret[3]  # aspect_ratio = width / height
        return ret
    
    @staticmethod
    def _tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr"""
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def tlwh(self):
        """Get current position in tlwh format (top left x, y, width, height)"""
        if self.mean is not None:
            # Convert from Kalman state (xyah) to tlwh
            ret = self.mean[:4].copy()
            ret[2] *= ret[3]  # width = aspect_ratio * height
            ret[:2] -= ret[2:] / 2  # top_left = center - width/height / 2
            return ret
        else:
            # Fallback to bbox
            return self._bbox_to_tlwh(self.bbox)
    
    @property
    def tlbr(self):
        """Get current position in tlbr format (top left x, y, bottom right x, y)"""
        tlwh = self.tlwh
        return self._tlwh_to_tlbr(tlwh)
    
    def to_xyah(self):
        """Convert to xyah format (for Kalman Filter)"""
        return self._tlwh_to_xyah(self.tlwh)
    
    def update(self, detection: Dict, features: np.ndarray, frame_id: int):
        """Update track with new detection (ByteTrack-compatible)"""
        # Convert detection bbox to xyah for Kalman Filter
        bbox = detection.get('bbox', self.bbox)
        tlwh = self._bbox_to_tlwh(bbox)
        xyah = self._tlwh_to_xyah(tlwh)
        
        # Update Kalman Filter (ByteTrack format)
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, xyah
            )
        else:
            # Initialize if not set
            self.mean, self.covariance = self.kalman_filter.initiate(xyah)
        
        # Update motion model (legacy, untuk backward compatibility)
        if len(self.history) >= 2:
            prev_center = self.current_detection.get('center', self.center)
            new_center = detection.get('center', self._bbox_to_center(bbox))
            new_velocity = np.array([
                new_center[0] - prev_center[0],
                new_center[1] - prev_center[1]
            ])
            alpha = 0.7
            self.velocity = alpha * self.velocity + (1 - alpha) * new_velocity
        
        # Update state
        self.current_detection = detection
        if features is not None:
            self.current_features = features
            self.feature_history.append(features)
        self.current_frame = frame_id
        self.last_seen = frame_id
        
        # Update history
        self.history.append(detection)
        self.frame_history.append(frame_id)
        
        # Update confidence tracking
        current_conf = detection.get('confidence', detection.get('score', 0.8))
        self.confidence_history.append(current_conf)
        self.avg_confidence = np.mean(self.confidence_history[-5:])
        
        # Update stability score
        self._update_stability_score()
        
        # Update statistics
        self.age += 1
        self.hits += 1
        self.misses = 0
        self.time_since_update = 0
        self.tracklet_len += 1
        self.state = 'active'
        self.is_activated = True
        
        # Update ByteTrack attributes
        self.bbox = bbox
        self.center = detection.get('center', self._bbox_to_center(bbox))
    
    def re_activate(self, detection: Dict, frame_id: int, new_id: bool = False):
        """
        Re-activate a lost track (ByteTrack-style)
        
        Args:
            detection: Detection dictionary
            frame_id: Current frame ID
            new_id: If True, assign new track ID. If False, keep existing ID (for recovery)
        """
        # Convert detection bbox to xyah for Kalman Filter
        bbox = detection.get('bbox', self.bbox)
        tlwh = self._bbox_to_tlwh(bbox)
        xyah = self._tlwh_to_xyah(tlwh)
        
        # Update Kalman Filter (ByteTrack format)
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, xyah
            )
        else:
            # Initialize if not set
            self.mean, self.covariance = self.kalman_filter.initiate(xyah)
        
        # Update state to active
        self.state = 'active'
        self.is_activated = True
        self.tracklet_len = 0  # Reset tracklet_len for re-activated tracks
        self.misses = 0
        self.time_since_update = 0
        
        # Assign new ID if requested (usually False for recovery)
        if new_id:
            # This would require access to next_track_id, but typically we want to keep ID
            pass  # Keep existing ID for recovery
        
        # Update detection and history
        self.current_detection = detection
        self.current_frame = frame_id
        self.last_seen = frame_id
        self.history.append(detection)
        self.frame_history.append(frame_id)
        
        # Update confidence tracking
        current_conf = detection.get('confidence', detection.get('score', 0.8))
        self.confidence_history.append(current_conf)
        self.avg_confidence = np.mean(self.confidence_history[-5:]) if len(self.confidence_history) > 0 else current_conf
        
        # Update ByteTrack attributes
        self.bbox = bbox
        self.center = detection.get('center', self._bbox_to_center(bbox))
        
        logger.debug(f"Track {self.track_id} re-activated (new_id={new_id})")
        self.confidence = current_conf
        self.score = current_conf
        
        # Reset ghost state
        if self.state in ['ghost', 'lost']:
            self.ghost_start_frame = None
            self.ghost_predictions = []
            self.ghost_confidence = 1.0
    
    def predict(self):
        """Predict next state using Kalman Filter (ByteTrack-compatible)"""
        if self.mean is not None and self.covariance is not None:
            # Set velocity to 0 if not tracked
            mean_state = self.mean.copy()
            if self.state != 'active' and self.state != 'tracked':
                mean_state[7] = 0  # Set velocity h to 0
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
    
    @staticmethod
    def multi_predict(tracks):
        """Vectorized prediction for multiple tracks (ByteTrack-compatible)"""
        if len(tracks) == 0:
            return
        
        # Collect all means and covariances
        multi_mean = np.asarray([t.mean.copy() for t in tracks if t.mean is not None])
        multi_covariance = np.asarray([t.covariance for t in tracks if t.covariance is not None])
        
        if len(multi_mean) == 0:
            return
        
        # Set velocity to 0 for non-tracked tracks
        for i, track in enumerate(tracks):
            if track.mean is not None:
                if track.state != 'active' and track.state != 'tracked':
                    multi_mean[i][7] = 0
        
        # Use shared Kalman filter instance for multi_predict
        shared_kf = KalmanFilter()
        multi_mean, multi_covariance = shared_kf.multi_predict(multi_mean, multi_covariance)
        
        # Update track states
        valid_idx = 0
        for track in tracks:
            if track.mean is not None:
                track.mean = multi_mean[valid_idx]
                track.covariance = multi_covariance[valid_idx]
                valid_idx += 1
    
    def _update_stability_score(self):
        """Calculate track stability based on motion consistency"""
        if len(self.history) < 3:
            self.stability_score = 1.0
            return
        
        recent_positions = [det['center'] for det in self.history[-5:]]
        if len(recent_positions) < 3:
            return
        
        velocities = []
        for i in range(1, len(recent_positions)):
            vel = [
                recent_positions[i][0] - recent_positions[i-1][0],
                recent_positions[i][1] - recent_positions[i-1][1]
            ]
            velocities.append(vel)
        
        if len(velocities) >= 2:
            velocity_std = np.std(velocities, axis=0)
            velocity_consistency = 1.0 / (1.0 + np.mean(velocity_std))
            self.stability_score = 0.7 * velocity_consistency + 0.3 * self.avg_confidence
    
    def miss(self, frame_id: int, enable_ghost: bool = True):
        """Mark track as missed in current frame"""
        old_state = self.state
        self.age += 1
        self.misses += 1
        self.time_since_update += 1
        self.current_frame = frame_id
        
        if enable_ghost and self.state == 'active':
            if self.hits >= 3 and self.stability_score > 0.5:
                self.state = 'ghost'
                self.ghost_start_frame = frame_id
                self.ghost_confidence = min(0.9, self.stability_score)
                logger.debug(f"[FRAME {frame_id}] Track {self.track_id} state: active → GHOST (hits={self.hits}, stability={self.stability_score:.2f}, misses={self.misses})")
            else:
                self.state = 'lost'
                logger.debug(f"[FRAME {frame_id}] Track {self.track_id} state: active → LOST (hits={self.hits}, stability={self.stability_score:.2f}, misses={self.misses})")
        elif self.state == 'ghost':
            self.ghost_confidence *= 0.8
            logger.debug(f"[FRAME {frame_id}] Track {self.track_id} still GHOST (misses={self.misses}, confidence={self.ghost_confidence:.2f})")
        elif self.state == 'lost':
            logger.debug(f"[FRAME {frame_id}] Track {self.track_id} still LOST (misses={self.misses})")
            
        # Termination conditions
        max_missing = 20 if self.hits >= 5 else 15
        if self.misses >= max_missing:
            self.state = 'terminated'
            logger.debug(f"[FRAME {frame_id}] Track {self.track_id} TERMINATED (misses={self.misses} >= {max_missing})")
        elif not enable_ghost and self.state != 'ghost':
            self.state = 'lost'
    
    def get_average_features(self, window_size: int = 5) -> np.ndarray:
        """Get average features over recent frames"""
        if not self.feature_history:
            return np.zeros(128, dtype=np.float32)
        
        recent_features = self.feature_history[-window_size:]
        if len(recent_features) == 1:
            return recent_features[0]
        
        weights = np.exp(np.linspace(-1, 0, len(recent_features)))
        weights = weights / np.sum(weights)
        
        weighted_features = np.zeros_like(recent_features[0])
        for i, features in enumerate(recent_features):
            weighted_features += weights[i] * features
        
        return weighted_features


class PureSmartHungarianTrackManager:
    """
    ByteTrack + Ghost Track TrackManager

    TrackManager menggunakan ByteTrack three-stage Hungarian dengan ghost track recovery:
    - Detection Pipeline (YOLOv8)
    - IoU Calculation Pipeline
    - ByteTrack Three-Stage Association:
      * Stage 1: High-confidence matching (conf >= 0.5)
      * Stage 2: Low-confidence recovery (0.1 <= conf < 0.5)
      * Stage 3: Ghost track recovery (NOVELTY!)
    - Track Management Pipeline

    SMART HUNGARIAN REMOVED - Pure ByteTrack approach
    """
    
    def __init__(self, config=None):
        """Initialize ByteTrack + Ghost Track TrackManager"""
        from utils.settings import SETTINGS

        # Use centralized settings
        if config is None:
            tracking_config = SETTINGS.get_tracking_config()
        else:
            if hasattr(config, 'tracking'):
                tracking_config = config.tracking.__dict__
            else:
                tracking_config = config

        # Basic tracking parameters
        self.reid_threshold = tracking_config['reid_threshold']
        self.max_missing_frames = tracking_config['max_missing_frames']
        self.min_track_length = tracking_config['min_track_length']

        # Track storage
        self.tracks = {}
        self.next_track_id = 1
        self.confirmed_tracks = {}

        # === INITIALIZE PIPELINE COMPONENTS ===
        print("Initializing ByteTrack + Ghost Track TrackManager pipeline components...")
        print("⚠️  SMART HUNGARIAN REMOVED - Using pure ByteTrack approach")
        
        # Detection Pipeline Component
        self.detector = YOLOv8Detector()
        print("✓ YOLOv8 Detection Pipeline initialized")
        
        # IoU Calculation Pipeline Component
        self.iou_calculator = IoUCalculator()
        print("✓ IoU Calculator Pipeline initialized")
        
        # ByteTrack-style Hungarian Matcher (MAIN ALGORITHM)
        from models.bytetrack_hungarian import ByteTrackHungarianMatcher
        bytetrack_config = {
            'stage1_conf_threshold': 0.5,
            'stage1_match_thresh': 0.7,  # ⚠️ TURUNKAN dari 0.8 ke 0.7 untuk recover lebih banyak lost tracks
            'stage2_conf_min': 0.1,
            'stage2_conf_max': 0.5,
            'stage2_match_thresh': 0.5,  # ✅ Kembali ke 0.5 (ByteTrack default)
            'unconfirmed_match_thresh': 0.7,  # ✅ Kembali ke 0.7 (ByteTrack default)
            'ghost_match_thresh': 0.5,  # ⚠️ Turunkan untuk recovery lebih agresif
            'use_score_fusion': True,  # ✅ Enable score fusion
            'min_iou_thresh': 0.25  # ⚠️ Turunkan sedikit untuk lebih lenient (0.3 -> 0.25)
        }
        self.bytetrack_matcher = ByteTrackHungarianMatcher(bytetrack_config)
        print("✓ ByteTrack Hungarian Matcher initialized (with score fusion)")
        print("✓ Smart Hungarian REMOVED - Using pure ByteTrack approach")
        
        # Shared Kalman Filter instance (ByteTrack-style)
        self.kalman_filter = KalmanFilter()
        
        # Statistics
        self.total_tracks_created = 0
        self.total_associations = 0
        self.total_id_switches = 0
        self.total_ghost_reidentifications = 0
        
        # Performance tracking
        self.frame_count = 0
        self.pipeline_stats = {
            'detection_time': [],
            'iou_calculation_time': [],
            'data_association_time': [],
            'track_update_time': [],
            'total_pipeline_time': []
        }
        
        print("🎯 ByteTrack + Ghost Track TrackManager initialized")
        print("   Pipeline: YOLOv8 → IoU Calculator → ByteTrack Three-Stage → TrackUpdate")
        print("   Algorithm: ByteTrack (High + Low) + Ghost Recovery (NOVELTY)")

    def process_frame(self, image: np.ndarray, frame_id: int) -> Dict:
        """
        MAIN PIPELINE METHOD - Process satu frame lengkap dengan ByteTrack Three-Stage
        """
        self.frame_count += 1
        pipeline_start_time = time.time()

        logger.info(f"Processing frame {frame_id} - ByteTrack Three-Stage Pipeline")
        
        # === DETECTION PIPELINE ===
        detection_start = time.time()
        detections = self._detection_pipeline(image)
        detection_time = time.time() - detection_start
        self.pipeline_stats['detection_time'].append(detection_time)
        
        # === IoU CALCULATION PIPELINE ===
        # Note: IoU calculation sekarang dilakukan di dalam matching module
        # Tidak perlu pre-calculate cost_matrix
        iou_start = time.time()
        iou_features = {}  # Placeholder, tidak digunakan lagi
        iou_time = time.time() - iou_start
        self.pipeline_stats['iou_calculation_time'].append(iou_time)
        
        # === BYTETRACK THREE-STAGE DATA ASSOCIATION PIPELINE ===
        association_start = time.time()

        logger.info("Using ByteTrack Three-Stage Association (High-conf + Low-conf + Ghost)")
        association_results = self._bytetrack_three_stage_association(
            detections, iou_features, frame_id
        )

        association_time = time.time() - association_start
        self.pipeline_stats['data_association_time'].append(association_time)
        
        # === TRACK UPDATE PIPELINE ===
        update_start = time.time()
        update_results = self._track_update_pipeline(association_results, detections, iou_features, frame_id)
        update_time = time.time() - update_start
        self.pipeline_stats['track_update_time'].append(update_time)
        
        # Total pipeline time
        total_pipeline_time = time.time() - pipeline_start_time
        self.pipeline_stats['total_pipeline_time'].append(total_pipeline_time)
        
        # Compile complete results
        pipeline_results = {
            'frame_id': frame_id,
            'detections': detections,
            'iou_features': iou_features,
            'active_tracks': self.get_current_tracks(),
            'association_results': association_results,
            'update_results': update_results,
            'pipeline_timing': {
                'detection_time': detection_time,
                'iou_calculation_time': iou_time,
                'data_association_time': association_time,
                'track_update_time': update_time,
                'total_pipeline_time': total_pipeline_time
            },
            'pipeline_stats': self._get_pipeline_stats()
        }
        
        logger.info(f"Frame {frame_id} processed: {len(detections)} detections → {len(self.get_current_tracks())} tracks")
        
        return pipeline_results
    
    def _detection_pipeline(self, image: np.ndarray) -> List[Dict]:
        """Detection Pipeline - YOLOv8 pedestrian detection"""
        logger.debug("Running Detection Pipeline (YOLOv8)")
        
        detections = self.detector.detect(image)
        
        logger.debug(f"Detection Pipeline: {len(detections)} detections found")
        return detections
    
    # ✅ Smart Hungarian methods REMOVED - using ByteTrack instead
    # Functions _pure_smart_hungarian_association_pipeline and _pure_smart_hungarian_association
    # have been completely removed. Use _bytetrack_three_stage_association instead.

    def _bytetrack_three_stage_association(self, detections: List[Dict], iou_features: Dict, frame_id: int) -> Dict:
        """
        ByteTrack-inspired Three-Stage Association Pipeline

        Stage 1: High-confidence detections (conf >= 0.5) with tracked tracks
        Stage 2: Low-confidence detections (0.1 <= conf < 0.5) with unmatched tracked tracks
        Stage 3: Ghost track recovery with remaining detections
        Unconfirmed: Track baru dengan threshold lebih ketat

        ✅ Key changes:
        - Kalman prediction SEBELUM matching
        - Score fusion untuk Stage 1
        - API baru: tidak perlu pass cost_matrix
        """
        logger.debug("Running ByteTrack Three-Stage Association Pipeline")

        if len(detections) == 0:
            return self._handle_empty_detections(frame_id)

        # Separate tracks by state
        tracked_tracks = [t for t in self.tracks.values() if t.state == 'active' and t.is_activated]
        lost_tracks = [t for t in self.tracks.values() if t.state == 'lost']  # ⭐ TAMBAHKAN LOST TRACKS
        unconfirmed_tracks = [t for t in self.tracks.values() if t.state == 'active' and not t.is_activated]
        ghost_tracks = [t for t in self.tracks.values() if t.state == 'ghost']

        if len(tracked_tracks) == 0 and len(lost_tracks) == 0 and len(unconfirmed_tracks) == 0 and len(ghost_tracks) == 0:
            return self._handle_no_tracks(detections, iou_features, frame_id)

        # === KALMAN PREDICTION (CRITICAL: Before matching!) ===
        # ⭐ Stage 1: tracked + lost (seperti ByteTrack asli)
        strack_pool = tracked_tracks + lost_tracks
        all_tracks_for_prediction = strack_pool + unconfirmed_tracks
        if len(all_tracks_for_prediction) > 0:
            Track.multi_predict(all_tracks_for_prediction)
            logger.debug(f"Kalman prediction: {len(all_tracks_for_prediction)} tracks predicted (tracked={len(tracked_tracks)}, lost={len(lost_tracks)}, unconfirmed={len(unconfirmed_tracks)})")

        # === STAGE 1: High-Confidence Matching (Tracked + Lost tracks, seperti ByteTrack asli) ===
        stage1_matches = []
        unmatched_strack_pool_idx = []
        unmatched_detections_idx = []
        unmatched_tracked_idx = []
        unmatched_lost_idx = []

        if len(strack_pool) > 0:
            logger.debug(f"Stage 1: Matching {len(strack_pool)} tracks (tracked={len(tracked_tracks)}, lost={len(lost_tracks)}) with high-conf detections")

            # ⭐ ByteTrack asli: Match tracked + lost tracks BERSAMA dengan threshold sama (0.8) dan score fusion
            matches_s1, unmatched_strack_pool_idx, unmatched_detections_idx = self.bytetrack_matcher.match_stage1(
                strack_pool, detections
            )

            # Convert to association format (dengan track state info)
            lost_recovered = 0
            tracked_matched = 0
            for track_idx, det_idx in matches_s1:
                track = strack_pool[track_idx]
                track_state = track.state
                
                # ⭐ DEBUG: Track recovery statistics
                if track_state == 'lost':
                    lost_recovered += 1
                    logger.debug(f"[FRAME {frame_id}] ⭐ Stage 1: Lost track {track.track_id} matched with det {det_idx} (will re-activate, misses={track.misses})")
                elif track_state == 'active':
                    tracked_matched += 1
                
                stage1_matches.append({
                    'track_id': track.track_id,
                    'detection_idx': det_idx,
                    'type': 'bytetrack_stage1_high_conf',
                    'stage': 1,
                    'track_state': track_state  # ⭐ Simpan state untuk handle re_activate
                })

            logger.debug(f"[FRAME {frame_id}] Stage 1: {len(stage1_matches)} matches (tracked={tracked_matched}, lost_recovered={lost_recovered}), {len(unmatched_strack_pool_idx)} unmatched")
            
            # Separate unmatched tracked vs lost untuk Stage 2
            unmatched_tracked_idx = []
            unmatched_lost_idx = []
            for idx in unmatched_strack_pool_idx:
                if idx < len(tracked_tracks):
                    unmatched_tracked_idx.append(idx)
                else:
                    # Lost track index (relative to strack_pool, need to map back)
                    lost_idx = idx - len(tracked_tracks)
                    unmatched_lost_idx.append(lost_idx)
                    logger.debug(f"[FRAME {frame_id}] Lost track {lost_tracks[lost_idx].track_id} unmatched in Stage 1 (misses={lost_tracks[lost_idx].misses})")
        else:
            unmatched_detections_idx = list(range(len(detections)))
            unmatched_tracked_idx = []
            unmatched_lost_idx = []

        # === STAGE 2: Low-Confidence Recovery (Tracked tracks only, bukan Lost) ===
        stage2_matches = []

        if len(unmatched_tracked_idx) > 0 and len(unmatched_detections_idx) > 0:
            logger.debug(f"Stage 2: Matching {len(unmatched_tracked_idx)} unmatched tracked tracks with low-conf detections")

            # Get unmatched tracked tracks (hanya yang tracked, bukan lost)
            unmatched_tracked = [tracked_tracks[i] for i in unmatched_tracked_idx]

            # Run Stage 2: Low-confidence matching (hanya tracked tracks, sesuai ByteTrack asli)
            matches_s2, still_unmatched_tracked, still_unmatched_dets = self.bytetrack_matcher.match_stage2(
                tracked_tracks,  # All tracked tracks (indices relative to tracked_tracks)
                detections,
                unmatched_tracked_idx,  # Indices of unmatched tracked tracks
                unmatched_detections_idx  # Available detection indices
            )

            # Convert to association format
            for track_idx, det_idx in matches_s2:
                stage2_matches.append({
                    'track_id': tracked_tracks[track_idx].track_id,
                    'detection_idx': det_idx,
                    'type': 'bytetrack_stage2_low_conf',
                    'stage': 2
                })

            # Update unmatched lists
            unmatched_tracked_idx = still_unmatched_tracked
            unmatched_detections_idx = still_unmatched_dets

            logger.debug(f"Stage 2: {len(stage2_matches)} matches, {len(unmatched_tracked_idx)} still unmatched")
        
        # Note: Lost tracks yang unmatched tidak masuk Stage 2 (sesuai ByteTrack asli)
        # Mereka akan tetap lost atau di-terminate

        # === UNCONFIRMED TRACKS MATCHING (ByteTrack-style) ===
        unconfirmed_matches = []
        unmatched_unconfirmed = []

        if len(unconfirmed_tracks) > 0 and len(unmatched_detections_idx) > 0:
            logger.debug(f"Unconfirmed: Matching {len(unconfirmed_tracks)} unconfirmed tracks")

            # Predict unconfirmed tracks juga
            Track.multi_predict(unconfirmed_tracks)

            # Match unconfirmed dengan threshold lebih ketat (0.7)
            matches_unconfirmed, unmatched_unconfirmed_idx, unmatched_dets_after_unconfirmed = self.bytetrack_matcher.match_unconfirmed(
                unconfirmed_tracks, detections
            )

            # Convert to association format
            for track_idx, det_idx in matches_unconfirmed:
                unconfirmed_matches.append({
                    'track_id': unconfirmed_tracks[track_idx].track_id,
                    'detection_idx': det_idx,
                    'type': 'bytetrack_unconfirmed',
                    'stage': 0  # Before stage 1
                })

            # Update unmatched detections
            unmatched_detections_idx = unmatched_dets_after_unconfirmed
            unmatched_unconfirmed = [unconfirmed_tracks[i] for i in unmatched_unconfirmed_idx]

            logger.debug(f"Unconfirmed: {len(unconfirmed_matches)} matches, {len(unmatched_unconfirmed)} unmatched")

        # === STAGE 3: Ghost Track Recovery ===
        stage3_matches = []

        if len(ghost_tracks) > 0 and len(unmatched_detections_idx) > 0:
            logger.debug(f"Stage 3: Ghost recovery - {len(ghost_tracks)} ghosts, {len(unmatched_detections_idx)} available dets")

            # Run Stage 3: Ghost track recovery
            matches_s3, unmatched_ghosts, final_unmatched_dets = self.bytetrack_matcher.match_ghost_tracks(
                ghost_tracks,
                detections,
                unmatched_detections_idx
            )

            # Convert to association format
            for ghost_idx, det_idx in matches_s3:
                stage3_matches.append({
                    'track_id': ghost_tracks[ghost_idx].track_id,
                    'detection_idx': det_idx,
                    'type': 'bytetrack_stage3_ghost_recovery',
                    'stage': 3,
                    'track_state': 'ghost'  # ⭐ Simpan state untuk handle re_activate
                })

            # Update unmatched detections
            unmatched_detections_idx = final_unmatched_dets

            # Unmatched ghost tracks
            unmatched_ghost_tracks = [ghost_tracks[i] for i in unmatched_ghosts]

            logger.debug(f"Stage 3: {len(stage3_matches)} ghost recoveries, {len(unmatched_ghosts)} ghosts remain")
        else:
            unmatched_ghost_tracks = ghost_tracks

        # === MERGE ALL STAGES ===
        all_associations = unconfirmed_matches + stage1_matches + stage2_matches + stage3_matches

        # Unmatched tracked tracks
        unmatched_tracked_tracks = [tracked_tracks[i] for i in unmatched_tracked_idx]
        
        # Unmatched lost tracks (tetap lost, tidak di-recover)
        unmatched_lost_tracks = [lost_tracks[i] for i in unmatched_lost_idx] if 'unmatched_lost_idx' in locals() else []

        # All unmatched tracks
        all_unmatched_tracks = unmatched_tracked_tracks + unmatched_lost_tracks + unmatched_unconfirmed + unmatched_ghost_tracks

        logger.info(f"ByteTrack Three-Stage Complete:")
        logger.info(f"  Unconfirmed: {len(unconfirmed_matches)} matches")
        logger.info(f"  Stage 1 (High-conf): {len(stage1_matches)} matches")
        logger.info(f"  Stage 2 (Low-conf): {len(stage2_matches)} matches")
        logger.info(f"  Stage 3 (Ghost): {len(stage3_matches)} matches")
        logger.info(f"  Total: {len(all_associations)} associations")
        logger.info(f"  Unmatched tracks: {len(all_unmatched_tracks)}, Unmatched detections: {len(unmatched_detections_idx)}")
        
        # ⭐ DEBUG: Summary statistics
        logger.debug(f"[FRAME {frame_id}] Summary: Unmatched tracked={len(unmatched_tracked_tracks)}, Unmatched lost={len(unmatched_lost_tracks)}, Unmatched ghost={len(unmatched_ghost_tracks)}")

        return {
            'associations': all_associations,
            'unmatched_tracks': all_unmatched_tracks,
            'unmatched_detections': unmatched_detections_idx,
            'active_tracks': tracked_tracks + unconfirmed_tracks + ghost_tracks,
            'stage_stats': {
                'unconfirmed_matches': len(unconfirmed_matches),
                'stage1_matches': len(stage1_matches),
                'stage2_matches': len(stage2_matches),
                'stage3_matches': len(stage3_matches),
                'total_matches': len(all_associations)
            }
        }

    def _low_confidence_recovery(self, unmatched_tracks: List[Track], detections: List[Dict],
                                main_associations: List[Dict], frame_id: int) -> List[Dict]:
        """
        Low-Confidence Recovery Stage

        Recover unmatched tracks menggunakan low-confidence detections (0.2-0.4)
        yang diabaikan di main matching stage.

        Strategy:
        - Pakai Kalman prediction untuk matching (bukan current position)
        - Lenient threshold karena detection quality rendah
        - Prioritas: recover track continuity > precision
        """
        recovery_matches = []

        if not unmatched_tracks or not detections:
            return recovery_matches

        # Get detections yang sudah dipakai di main stage
        used_det_indices = set([assoc['detection_idx'] for assoc in main_associations])

        # Filter low-confidence detections yang belum dipakai
        low_conf_detections = []
        for det_idx, detection in enumerate(detections):
            conf = detection.get('confidence', 0.8)
            if 0.2 <= conf < 0.4 and det_idx not in used_det_indices:
                low_conf_detections.append((det_idx, detection))

        if not low_conf_detections:
            logger.debug("No low-confidence detections available for recovery")
            return recovery_matches

        logger.debug(f"Low-conf recovery: {len(unmatched_tracks)} unmatched tracks, "
                    f"{len(low_conf_detections)} low-conf detections")

        # Match unmatched tracks dengan low-conf detections
        for track in unmatched_tracks:
            # Skip tracks yang baru dibuat (hits < 3)
            if track.hits < 3:
                continue

            # Predict position dengan Kalman (ByteTrack-style)
            # Track sudah di-predict di multi_predict sebelumnya, jadi kita bisa pakai predicted position
            # Atau predict lagi jika belum
            if track.mean is not None and track.covariance is not None:
                track.predict()  # Predict menggunakan internal state
                # Get predicted center dari tlwh
                predicted_tlwh = track.tlwh
                predicted_center = [predicted_tlwh[0] + predicted_tlwh[2]/2, 
                                   predicted_tlwh[1] + predicted_tlwh[3]/2]
            else:
                # Fallback ke current center jika Kalman belum initialized
                predicted_center = track.center

            best_match_idx = None
            best_match_det = None
            best_score = 0

            for det_idx, detection in low_conf_detections:
                # Get detection center
                if 'center' in detection:
                    det_center = detection['center']
                else:
                    # Calculate center from bbox
                    bbox = detection.get('bbox', [0, 0, 1, 1])
                    if len(bbox) == 4:
                        if bbox[2] < 100:  # (x, y, w, h)
                            det_center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
                        else:  # (x1, y1, x2, y2)
                            det_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                    else:
                        det_center = [0, 0]
                
                # Distance dari predicted position
                dist = np.sqrt(
                    (predicted_center[0] - det_center[0])**2 +
                    (predicted_center[1] - det_center[1])**2
                )

                # Lenient distance threshold (150 pixel)
                max_recovery_distance = 150.0

                if dist < max_recovery_distance:
                    # Score berdasarkan proximity
                    proximity_score = 1.0 - (dist / max_recovery_distance)

                    # Size consistency check
                    track_tlbr = track.tlbr  # Use tlbr property
                    det_bbox = detection.get('bbox', [0, 0, 1, 1])
                    
                    # Convert det_bbox ke tlbr jika belum
                    if len(det_bbox) == 4:
                        if det_bbox[2] < 100:  # Likely (x, y, w, h)
                            det_tlbr = [det_bbox[0], det_bbox[1], 
                                       det_bbox[0] + det_bbox[2], 
                                       det_bbox[1] + det_bbox[3]]
                        else:  # Already (x1, y1, x2, y2)
                            det_tlbr = det_bbox
                    else:
                        det_tlbr = [0, 0, 1, 1]

                    track_area = (track_tlbr[2] - track_tlbr[0]) * (track_tlbr[3] - track_tlbr[1])
                    det_area = (det_tlbr[2] - det_tlbr[0]) * (det_tlbr[3] - det_tlbr[1])

                    if track_area > 0 and det_area > 0:
                        size_ratio = min(track_area, det_area) / max(track_area, det_area)
                    else:
                        size_ratio = 0.5

                    # Combined score
                    # Proximity lebih penting (0.7) karena pakai prediction
                    # Size consistency sebagai sanity check (0.3)
                    combined_score = 0.7 * proximity_score + 0.3 * size_ratio

                    # Minimum size consistency requirement
                    if size_ratio > 0.4 and combined_score > best_score:
                        best_score = combined_score
                        best_match_idx = det_idx
                        best_match_det = detection

            # Accept match jika score cukup tinggi
            recovery_threshold = 0.5
            if best_match_det is not None and best_score > recovery_threshold:
                recovery_matches.append({
                    'track_id': track.track_id,
                    'detection_idx': best_match_idx,
                    'score': best_score,
                    'type': 'low_confidence_recovery',
                    'detection_confidence': best_match_det.get('confidence', 0)
                })

                # Remove detection dari available list
                low_conf_detections = [(idx, det) for idx, det in low_conf_detections
                                      if idx != best_match_idx]

                logger.debug(f"Recovery match: Track {track.track_id} ← Detection {best_match_idx} "
                           f"(score={best_score:.3f}, conf={best_match_det.get('confidence', 0):.2f})")

        logger.info(f"Low-conf recovery: {len(recovery_matches)} tracks recovered")

        return recovery_matches

    def _associate_ghost_tracks(self, ghost_tracks: List[Track], detections: List[Dict],
                              iou_features: Dict, available_detections: List[int]) -> Dict:
        """Ghost track association menggunakan IoU features"""
        associations = []
        matched_detections = []
        unmatched_ghost_tracks = []
        
        # Get IoU matrix from features
        iou_matrix = iou_features.get('iou_matrix', np.array([]))
        
        if iou_matrix.size == 0:
            return {
                'associations': [],
                'matched_detections': [],
                'unmatched_ghost_tracks': ghost_tracks
            }
        
        for ghost_track in ghost_tracks:
            best_match = None
            best_iou = 0
            best_det_idx = -1
            
            # Find best IoU match for this ghost track
            ghost_track_idx = None
            for i, track in enumerate(self.tracks.values()):
                if track.track_id == ghost_track.track_id:
                    ghost_track_idx = i
                    break
            
            if ghost_track_idx is not None and ghost_track_idx < iou_matrix.shape[0]:
                for det_idx in available_detections:
                    if det_idx < iou_matrix.shape[1] and det_idx < len(detections):
                        iou = iou_matrix[ghost_track_idx, det_idx]
                        
                        # Spatial constraint for ghost tracks
                        detection = detections[det_idx]
                        spatial_distance = np.sqrt(
                            (ghost_track.center[0] - detection['center'][0])**2 +
                            (ghost_track.center[1] - detection['center'][1])**2
                        )
                        
                        max_ghost_distance = 150.0
                        min_ghost_iou = 0.3
                        
                        if (iou > best_iou and 
                            iou > min_ghost_iou and
                            spatial_distance < max_ghost_distance):
                            
                            best_iou = iou
                            best_match = detection
                            best_det_idx = det_idx
            
            if best_match is not None:
                associations.append({
                    'track_id': ghost_track.track_id,
                    'detection_idx': best_det_idx,
                    'cost': 1.0 - best_iou,
                    'type': 'ghost_reidentification_iou',
                    'iou': best_iou
                })
                matched_detections.append(best_det_idx)
                self.total_ghost_reidentifications += 1
            else:
                unmatched_ghost_tracks.append(ghost_track)
        
        return {
            'associations': associations,
            'matched_detections': matched_detections,
            'unmatched_ghost_tracks': unmatched_ghost_tracks
        }
    
    def _track_update_pipeline(self, association_results: Dict, detections: List[Dict],
                             iou_features: Dict, frame_id: int) -> Dict:
        """Track Update Pipeline - Update track states dan lifecycle management"""
        logger.debug("Running Track Update Pipeline")

        ghost_reidentifications = 0
        bytetrack_associations = 0
        low_conf_recoveries = 0

        # Update associated tracks
        for assoc in association_results['associations']:
            track_id = assoc['track_id']
            det_idx = assoc['detection_idx']

            track = self.tracks[track_id]
            detection = detections[det_idx]

            # Count association types
            assoc_type = assoc.get('type', '')
            if 'ghost' in assoc_type.lower():
                ghost_reidentifications += 1
            elif 'stage' in assoc_type or 'bytetrack' in assoc_type.lower():
                bytetrack_associations += 1

            # ⭐ Handle re-activation untuk lost tracks (ByteTrack-style)
            track_state = assoc.get('track_state', track.state)
            old_id = track.track_id  # ⭐ DEBUG: Track ID before update
            
            if track_state == 'lost':
                # Re-activate lost track dengan ID yang SAMA (new_id=False)
                track.re_activate(detection, frame_id, new_id=False)
                logger.debug(f"[FRAME {frame_id}] ⭐ RE-ACTIVATE: Track {old_id} re-activated from LOST state (ID preserved, misses={track.misses})")
            elif track_state == 'ghost':
                # Re-activate ghost track
                track.re_activate(detection, frame_id, new_id=False)
                track.state = 'active'  # Change from ghost to active
                logger.debug(f"[FRAME {frame_id}] ⭐ RE-ACTIVATE: Track {old_id} re-activated from GHOST state (ID preserved)")
            else:
                # Normal update untuk tracked/unconfirmed tracks
                track.update(detection, None, frame_id)
                logger.debug(f"[FRAME {frame_id}] Track {old_id} updated (state={track_state}, hits={track.hits})")
            
            # Activate track setelah match pertama (jika unconfirmed)
            if not track.is_activated and track_state != 'lost':
                track.is_activated = True
                logger.debug(f"Track {track.track_id} activated after first match")

        # === LOW-CONFIDENCE RECOVERY STAGE (DISABLED for now to reduce ID switches) ===
        # ⚠️ Low-confidence recovery dapat menyebabkan ID switches jika terlalu agresif
        # Disable sementara untuk reduce ID switches, bisa enable lagi jika perlu
        recovery_matches = []
        # recovery_matches = self._low_confidence_recovery(
        #     association_results['unmatched_tracks'],
        #     detections,
        #     association_results['associations'],
        #     frame_id
        # )

        # Update tracks dari recovery matches
        for match in recovery_matches:
            track = self.tracks[match['track_id']]
            detection = detections[match['detection_idx']]
            track.update(detection, None, frame_id)
            low_conf_recoveries += 1
            logger.debug(f"Low-conf recovery: Track {match['track_id']} recovered with conf={detection.get('confidence', 0):.2f}")

        # Update unmatched tracks (exclude yang sudah di-recover)
        recovered_track_ids = [m['track_id'] for m in recovery_matches]
        unmatched_tracks_final = [t for t in association_results['unmatched_tracks']
                                 if t.track_id not in recovered_track_ids]

        # Handle unmatched tracks (yang benar-benar tidak match)
        for track in unmatched_tracks_final:
            from utils.settings import SETTINGS
            enable_ghost = SETTINGS.get_tracking_config().get('enable_ghost_nodes', True)
            track.miss(frame_id, enable_ghost=enable_ghost)

        # Create new tracks (exclude detections yang sudah dipakai untuk recovery)
        used_det_indices = [m['detection_idx'] for m in association_results['associations']]
        used_det_indices.extend([m['detection_idx'] for m in recovery_matches])

        new_tracks = []
        for det_idx in association_results['unmatched_detections']:
            if det_idx in used_det_indices:
                continue

            detection = detections[det_idx]

            min_confidence = 0.4
            if detection.get('confidence', 0.8) >= min_confidence:
                new_track = self._create_new_track(detection, None, frame_id)
                new_tracks.append(new_track)
                logger.debug(f"[FRAME {frame_id}] ⭐ NEW TRACK CREATED: Track ID {new_track.track_id} from detection {det_idx} (conf={detection.get('confidence', 0):.2f})")

        # Cleanup terminated tracks
        terminated_tracks = self._cleanup_tracks()

        # Update confirmed tracks
        self._update_confirmed_tracks()

        logger.debug(f"Track Update Pipeline: {len(association_results['associations'])} updated, "
                    f"{len(recovery_matches)} recovered, {len(new_tracks)} new, {len(terminated_tracks)} terminated")

        return {
            'updated_tracks': len(association_results['associations']),
            'recovered_tracks': len(recovery_matches),
            'new_tracks': len(new_tracks),
            'terminated_tracks': len(terminated_tracks),
            'ghost_reidentifications': ghost_reidentifications,
            'bytetrack_associations': bytetrack_associations,
            'low_confidence_recoveries': low_conf_recoveries,
            'ghost_associations': association_results.get('ghost_associations', 0),
            'total_active_tracks': len([t for t in self.tracks.values() if t.state == 'active']),
            'total_ghost_tracks': len([t for t in self.tracks.values() if t.state == 'ghost']),
            'bytetrack_info': association_results.get('stage_stats', {})
        }
    
    
    def _create_new_track(self, detection: Dict, features: Optional[np.ndarray], frame_id: int) -> Track:
        """Create new track (unconfirmed, akan di-activate setelah confirmed)"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        # Create track (is_activated=False untuk unconfirmed track)
        new_track = Track(track_id, detection, features, frame_id)
        new_track.is_activated = False  # Unconfirmed track baru
        self.tracks[track_id] = new_track
        
        self.total_tracks_created += 1
        
        logger.debug(f"Created new unconfirmed track {track_id}")
        return new_track
    
    def _cleanup_tracks(self) -> List[Track]:
        """Remove terminated tracks"""
        terminated_tracks = []
        
        track_ids_to_remove = []
        for track_id, track in self.tracks.items():
            if track.state == 'terminated':
                terminated_tracks.append(track)
                track_ids_to_remove.append(track_id)
        
        for track_id in track_ids_to_remove:
            del self.tracks[track_id]
            if track_id in self.confirmed_tracks:
                del self.confirmed_tracks[track_id]
        
        return terminated_tracks
    
    def _update_confirmed_tracks(self):
        """Update confirmed tracks based on track quality"""
        for track_id, track in self.tracks.items():
            # Track dianggap confirmed jika:
            # - Sudah activated
            # - Hits >= min_track_length
            # - Stability score cukup baik
            if (track.is_activated and
                track.hits >= self.min_track_length and 
                getattr(track, 'stability_score', 0.5) > 0.6 and
                track_id not in self.confirmed_tracks):
                self.confirmed_tracks[track_id] = track
                logger.debug(f"Track {track_id} confirmed (hits={track.hits}, stability={getattr(track, 'stability_score', 0):.2f})")
    
    def _handle_empty_detections(self, frame_id: int) -> Dict:
        """Handle case with no detections"""
        unmatched_tracks = []
        
        for track in self.tracks.values():
            if track.state in ['active', 'ghost']:
                from utils.settings import SETTINGS
                enable_ghost = SETTINGS.get_tracking_config().get('enable_ghost_nodes', True)
                track.miss(frame_id, enable_ghost=enable_ghost)
                unmatched_tracks.append(track)
        
        return {
            'associations': [],
            'unmatched_tracks': unmatched_tracks,
            'unmatched_detections': [],
            'active_tracks': [],
            'ghost_associations': 0,
            'bytetrack_info': {'assignment_rate': 0.0, 'execution_time': 0.0, 'algorithm_used': 'no_detections'}
        }
    
    def _handle_no_tracks(self, detections: List[Dict], iou_features: Dict, frame_id: int) -> Dict:
        """Handle case with no existing tracks"""
        return {
            'associations': [],
            'unmatched_tracks': [],
            'unmatched_detections': list(range(len(detections))),
            'active_tracks': [],
            'ghost_associations': 0,
            'bytetrack_info': {'assignment_rate': 0.0, 'execution_time': 0.0, 'algorithm_used': 'no_tracks'}
        }
    
    def _get_pipeline_stats(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        if not self.pipeline_stats['total_pipeline_time']:
            return {'frames_processed': 0}
        
        recent_frames = 20  # Stats untuk 20 frames terakhir
        stats = {}
        
        for stage, times in self.pipeline_stats.items():
            recent_times = times[-recent_frames:] if len(times) > recent_frames else times
            if recent_times:
                stats[f'{stage}_avg'] = np.mean(recent_times)
                stats[f'{stage}_std'] = np.std(recent_times)
        
        # Calculate FPS
        recent_total_times = self.pipeline_stats['total_pipeline_time'][-recent_frames:]
        if recent_total_times:
            avg_total_time = np.mean(recent_total_times)
            stats['fps'] = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        stats['frames_processed'] = self.frame_count
        
        return stats
    
    # === PUBLIC INTERFACE METHODS ===
    
    def get_current_tracks(self, confirmed_only: bool = False) -> List[Track]:
        """Get current active tracks"""
        if confirmed_only:
            return [track for track in self.confirmed_tracks.values() 
                   if track.state == 'active']
        else:
            return [track for track in self.tracks.values() 
                   if track.state == 'active']
    
    def get_track_statistics(self) -> Dict:
        """Get comprehensive tracking statistics"""
        active_tracks = len([t for t in self.tracks.values() if t.state == 'active'])
        confirmed_tracks = len(self.confirmed_tracks)
        ghost_tracks = len([t for t in self.tracks.values() if t.state == 'ghost'])
        
        base_stats = {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': active_tracks,
            'ghost_tracks': ghost_tracks,
            'confirmed_tracks': confirmed_tracks,
            'total_associations': self.total_associations,
            'total_id_switches': self.total_id_switches,
            'total_ghost_reidentifications': self.total_ghost_reidentifications,
            'frames_processed': self.frame_count
        }
        
        # Add ByteTrack matching algorithm info
        base_stats.update({
            'matching_algorithm': 'bytetrack_hungarian',
            'pipeline_orchestrator': 'bytetrack_trackmanager'
        })
        
        # Add pipeline performance stats
        base_stats['pipeline_performance'] = self._get_pipeline_stats()
                
        return base_stats
    
    def get_bytetrack_performance_summary(self) -> Dict:
        """Get ByteTrack matching performance summary"""
        return {
            'available': True,
            'matching_algorithm': 'bytetrack_hungarian',
            'stage1_threshold': self.bytetrack_matcher.stage1_match_thresh,
            'stage2_threshold': self.bytetrack_matcher.stage2_match_thresh,
            'score_fusion': self.bytetrack_matcher.use_score_fusion,
            'architecture': 'bytetrack_three_stage',
            'orchestrator': 'bytetrack_trackmanager'
        }
    
    def visualize_tracks(self, image: np.ndarray, 
                        show_confirmed_only: bool = False,
                        show_trajectories: bool = True,
                        show_ghost_tracks: bool = True) -> np.ndarray:
        """Visualize tracks dengan Pure Smart Hungarian performance info"""
        vis_image = image.copy()
        
        tracks_to_show = self.get_current_tracks(confirmed_only=show_confirmed_only)
        ghost_tracks = [t for t in self.tracks.values() if t.state == 'ghost']
        
        # Draw active tracks
        for track in tracks_to_show:
            if track.state != 'ghost':
                color = self._get_track_color(track.track_id)
                
                bbox = track.current_detection['bbox']
                x1, y1, x2, y2 = bbox
                
                thickness = 3 if track.track_id in self.confirmed_tracks else 2
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                
                # Label dengan track info
                confidence = track.current_detection.get('confidence', 0.8)
                label = f"ID:{track.track_id} ({confidence:.2f}) H:{track.hits}"
                if hasattr(track, 'stability_score'):
                    label += f" S:{track.stability_score:.2f}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw trajectory
                if show_trajectories and len(track.history) > 1:
                    centers = [det['center'] for det in track.history[-10:]]
                    for i in range(1, len(centers)):
                        pt1 = (int(centers[i-1][0]), int(centers[i-1][1]))
                        pt2 = (int(centers[i][0]), int(centers[i][1]))
                        cv2.line(vis_image, pt1, pt2, color, 2)
                        cv2.circle(vis_image, pt2, 3, color, -1)
        
        # Enhanced legend dengan Pure Smart Hungarian info
        active_count = len([t for t in tracks_to_show if t.state == 'active'])
        confirmed_count = len([t for t in tracks_to_show if t.track_id in self.confirmed_tracks])
        ghost_count = len(ghost_tracks)
        
        info_text = f"BYTETRACK: Active:{active_count}, Confirmed:{confirmed_count}, Ghost:{ghost_count}"
        cv2.putText(vis_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Pipeline performance info
        pipeline_stats = self._get_pipeline_stats()
        fps = pipeline_stats.get('fps', 0)
        pipeline_text = f"Pipeline FPS: {fps:.1f}, Frames: {self.frame_count}"
        cv2.putText(vis_image, pipeline_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ByteTrack matching info
        hungarian_text = f"ByteTrack: Stage1={self.bytetrack_matcher.stage1_match_thresh}, Stage2={self.bytetrack_matcher.stage2_match_thresh}"
        cv2.putText(vis_image, hungarian_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Architecture info
        arch_text = f"YOLOv8 → ByteTrack Hungarian (3-Stage)"
        cv2.putText(vis_image, arch_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return vis_image
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        return color


# BACKWARD COMPATIBILITY & ALIASES
PureTrackManager = PureSmartHungarianTrackManager
PureSmartHungarianManager = PureSmartHungarianTrackManager


# TESTING
if __name__ == "__main__":
    print("🎯 Pure Smart Hungarian TrackManager")
    print("=" * 60)
    
    # Test initialization
    track_manager = PureSmartHungarianTrackManager()
    
    # Print pipeline info
    print(f"\n🔧 Pure Pipeline Architecture:")
    print(f"  Detection: YOLOv8Detector")
    print(f"  Feature Extraction: MobileNetV3Extractor")
    print(f"  Graph Construction: GraphBuilder")
    print(f"  GNN Prediction: GATTracker")
    print(f"  Data Association: Pure Smart Hungarian (Quality Controlled)")
    print(f"  Track Management: Enhanced lifecycle")
    
    # Create dummy frame untuk test
    dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    print(f"\n🧪 Testing pure pipeline with dummy frame...")
    
    try:
        # Test pipeline
        results = track_manager.process_frame(dummy_frame, frame_id=1)
        
        print(f"\n🎯 Pure Smart Hungarian Pipeline Results:")
        print(f"  Frame: {results['frame_id']}")
        print(f"  Detections: {len(results['detections'])}")
        print(f"  Active tracks: {len(results['active_tracks'])}")
        print(f"  Pipeline timing: {results['pipeline_timing']}")
        
        # Get statistics
        stats = track_manager.get_track_statistics()
        pure_hungarian_perf = track_manager.get_smart_hungarian_performance_summary()
        
        print(f"\n🎯 Pure Smart Hungarian Statistics:")
        print(f"  Total tracks created: {stats['total_tracks_created']}")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Pure Smart Hungarian success rate: {stats['smart_hungarian_success_rate']:.2%}")
        print(f"  Pipeline orchestrator: {stats['pipeline_orchestrator']}")
        
        if pure_hungarian_perf['available']:
            print(f"\n🚀 Pure Smart Hungarian Performance:")
            print(f"  Architecture: {pure_hungarian_perf['architecture']}")
            print(f"  Quality controlled assignments: {pure_hungarian_perf['quality_controlled_assignments']}")
        
        print("\n✅ Pure Smart Hungarian TrackManager test completed")
        print("🎯 Ready for pure implementation!")
        print("🔧 Evaluator usage: track_manager.process_frame(image, frame_id)")
        
    except Exception as e:
        print(f"\n❌ Pure Smart Hungarian pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 PURE IMPLEMENTATION SUMMARY:")
    print("1. NO AGM dependency - Pure Smart Hungarian only")
    print("2. Simplified uncertainty calculation with 5 factors:")
    print("   - IoU uncertainty")
    print("   - Distance uncertainty")  
    print("   - Feature similarity uncertainty")
    print("   - Motion prediction uncertainty")
    print("   - Confidence uncertainty")
    print("3. Smart Hungarian quality-controlled assignment")
    print("4. 5-layer quality gate system")
    print("5. Scene-adaptive thresholds")
    print("6. Expected improvements:")
    print("   - ID Switch: -30% hingga -50%")
    print("   - MOTA: maintained atau +1-3%")
    print("   - Simplified implementation")
    print("   - No complex AGM fallbacks")
    print("=" * 60)