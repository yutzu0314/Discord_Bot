"""
ByteTrack-style Hungarian Matcher
==================================

Pure Hungarian algorithm yang sesuai dengan ByteTrack asli.
Two-stage matching: high-conf + low-conf detections dengan score fusion.

Key features:
1. Score fusion: Menggabungkan IoU distance dengan detection confidence
2. Kalman prediction: Menggunakan predicted position untuk IoU calculation
3. Two-stage matching: High-conf (thresh 0.8) + Low-conf (thresh 0.5)
4. Unconfirmed track handling: Threshold lebih ketat untuk track baru

Author: ByteTrack Implementation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

from .matching import iou_distance, fuse_score, linear_assignment

logger = logging.getLogger(__name__)


class ByteTrackHungarianMatcher:
    """
    ByteTrack-inspired Hungarian matcher for two-stage association.

    Stage 1: Match tracks with high-confidence detections (conf >= 0.5)
    Stage 2: Match remaining tracks with low-confidence detections (0.1 <= conf < 0.5)

    Unlike Smart Hungarian, this does NOT reject assignments after Hungarian optimization.
    Instead, we filter the cost matrix BEFORE running Hungarian to ensure only valid
    assignments are considered.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize ByteTrack Hungarian matcher"""
        self.config = config or {}

        # Stage 1: High-confidence matching parameters (sesuai ByteTrack asli)
        self.stage1_conf_threshold = self.config.get('stage1_conf_threshold', 0.5)
        self.stage1_match_thresh = self.config.get('stage1_match_thresh', 0.8)  # ByteTrack default
        
        # ⭐ Lost track recovery: threshold lebih lenient untuk recovery
        self.lost_track_match_thresh = self.config.get('lost_track_match_thresh', 0.65)  # More lenient for lost tracks
        self.lost_track_use_score_fusion = self.config.get('lost_track_use_score_fusion', False)  # No fusion for lost tracks

        # Stage 2: Low-confidence matching parameters
        self.stage2_conf_min = self.config.get('stage2_conf_min', 0.1)
        self.stage2_conf_max = self.config.get('stage2_conf_max', 0.5)
        self.stage2_match_thresh = self.config.get('stage2_match_thresh', 0.5)  # ByteTrack default

        # Unconfirmed track matching (track baru)
        self.unconfirmed_match_thresh = self.config.get('unconfirmed_match_thresh', 0.7)
        
        # Ghost track matching
        self.ghost_match_thresh = self.config.get('ghost_match_thresh', 0.5)

        # Use score fusion (ByteTrack innovation)
        self.use_score_fusion = self.config.get('use_score_fusion', True)
        
        # Minimum IoU threshold untuk pre-filtering (reject matches yang jelas-jelas salah)
        self.min_iou_thresh = self.config.get('min_iou_thresh', 0.3)

        logger.info("ByteTrack Hungarian Matcher initialized (with score fusion)")
        logger.info(f"  Stage 1: conf >= {self.stage1_conf_threshold}, match_thresh = {self.stage1_match_thresh}")
        logger.info(f"  Stage 2: {self.stage2_conf_min} <= conf < {self.stage2_conf_max}, match_thresh = {self.stage2_match_thresh}")
        logger.info(f"  Score fusion: {self.use_score_fusion}")
        logger.info(f"  ⭐ Lost track recovery: match_thresh = {self.lost_track_match_thresh}, use_score_fusion = {self.lost_track_use_score_fusion}")

    def match_stage1_with_lost_tracks(self, tracked_tracks: List, lost_tracks: List, 
                                     detections: List[Dict]) -> Tuple[List[Tuple[int, int, str]], List[int], List[int], List[int]]:
        """
        Stage 1: Match tracked and lost tracks separately (lost tracks with more lenient threshold)
        
        Args:
            tracked_tracks: List of tracked Track objects
            lost_tracks: List of lost Track objects
            detections: List of detection dictionaries
            
        Returns:
            matches: List of (track_idx, det_idx, track_type) tuples where track_type is 'tracked' or 'lost'
            unmatched_tracked_idx: List of unmatched tracked track indices
            unmatched_lost_idx: List of unmatched lost track indices
            unmatched_detections: List of unmatched detection indices
        """
        all_matches = []
        unmatched_detections = []
        
        # Filter high-confidence detections
        high_conf_detections = []
        high_conf_indices = []
        for det_idx, detection in enumerate(detections):
            conf = detection.get('confidence', detection.get('score', 0.8))
            if conf >= self.stage1_conf_threshold:
                high_conf_detections.append(detection)
                high_conf_indices.append(det_idx)
        
        if len(high_conf_detections) == 0:
            logger.debug("Stage 1: No high-confidence detections")
            return [], list(range(len(tracked_tracks))), list(range(len(lost_tracks))), list(range(len(detections)))
        
        # === MATCH TRACKED TRACKS (threshold ketat 0.8, dengan score fusion) ===
        tracked_matches = []
        unmatched_tracked_idx = []
        if len(tracked_tracks) > 0:
            dists_tracked = iou_distance(tracked_tracks, high_conf_detections)
            
            # Pre-filtering untuk tracked tracks
            if self.min_iou_thresh > 0:
                min_cost = 1.0 - self.min_iou_thresh
                dists_tracked[dists_tracked > min_cost] = np.inf
            
            # Score fusion untuk tracked tracks
            if self.use_score_fusion:
                dists_tracked = fuse_score(dists_tracked, high_conf_detections)
            
            matches_tracked, unmatched_tracked, unmatched_dets_tracked = linear_assignment(
                dists_tracked, thresh=self.stage1_match_thresh
            )
            
            # Map detection indices
            for track_idx, det_idx in matches_tracked:
                tracked_matches.append((track_idx, high_conf_indices[det_idx], 'tracked'))
                all_matches.append((track_idx, high_conf_indices[det_idx], 'tracked'))
            
            unmatched_tracked_idx = unmatched_tracked
            unmatched_detections.extend([high_conf_indices[i] for i in unmatched_dets_tracked])
        else:
            unmatched_tracked_idx = []
            unmatched_detections = high_conf_indices.copy()
        
        # === MATCH LOST TRACKS (threshold lebih lenient 0.65, TANPA score fusion) ===
        lost_matches = []
        unmatched_lost_idx = []
        if len(lost_tracks) > 0 and len(unmatched_detections) > 0:
            # Only try to match with unmatched detections
            unmatched_detections_list = [detections[i] for i in unmatched_detections]
            
            dists_lost = iou_distance(lost_tracks, unmatched_detections_list)
            
            # ⭐ Pre-filtering lebih lenient untuk lost tracks (0.2 instead of 0.3)
            if self.min_iou_thresh > 0:
                lost_min_iou = max(0.2, self.min_iou_thresh - 0.1)  # More lenient
                min_cost = 1.0 - lost_min_iou
                dists_lost[dists_lost > min_cost] = np.inf
                logger.debug(f"Stage 1 Lost: Pre-filtered with min_iou={lost_min_iou} (more lenient)")
            
            # ⭐ TIDAK pakai score fusion untuk lost tracks (IoU murni)
            # Score fusion bisa membuat cost terlalu tinggi untuk recovery
            
            # ⭐ Threshold lebih lenient untuk lost tracks
            matches_lost, unmatched_lost, unmatched_dets_lost = linear_assignment(
                dists_lost, thresh=self.lost_track_match_thresh  # 0.65 instead of 0.8
            )
            
            # Map detection indices (relative to unmatched_detections)
            matched_det_indices_set = set()
            for lost_idx, det_idx_in_unmatched in matches_lost:
                actual_det_idx = unmatched_detections[det_idx_in_unmatched]
                lost_matches.append((lost_idx, actual_det_idx, 'lost'))
                all_matches.append((lost_idx, actual_det_idx, 'lost'))
                matched_det_indices_set.add(actual_det_idx)
            
            # Remove matched detections from unmatched_detections list
            unmatched_detections = [det_idx for det_idx in unmatched_detections if det_idx not in matched_det_indices_set]
            
            unmatched_lost_idx = unmatched_lost
            logger.debug(f"Stage 1 Lost: {len(lost_matches)} lost tracks recovered with threshold {self.lost_track_match_thresh}")
        else:
            unmatched_lost_idx = list(range(len(lost_tracks))) if len(lost_tracks) > 0 else []
        
        # Remove duplicates from unmatched_detections
        unmatched_detections = list(set(unmatched_detections))
        
        return all_matches, unmatched_tracked_idx, unmatched_lost_idx, unmatched_detections
    
    def match_stage1(self, tracks: List, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Stage 1: High-confidence matching (sesuai ByteTrack asli)

        Match active tracks with high-confidence detections using Hungarian algorithm.
        Menggunakan score fusion untuk menggabungkan IoU dengan detection confidence.

        Args:
            tracks: List of Track objects (harus sudah di-predict dengan Kalman)
            detections: List of detection dictionaries dengan confidence

        Returns:
            matches: List of (track_idx, det_idx) tuples
            unmatched_tracks: List of unmatched track indices
            unmatched_detections: List of unmatched detection indices
        """
        # Filter high-confidence detections
        high_conf_detections = []
        high_conf_indices = []
        for det_idx, detection in enumerate(detections):
            conf = detection.get('confidence', detection.get('score', 0.8))
            if conf >= self.stage1_conf_threshold:
                high_conf_detections.append(detection)
                high_conf_indices.append(det_idx)

        if len(high_conf_detections) == 0:
            logger.debug("Stage 1: No high-confidence detections")
            return [], list(range(len(tracks))), list(range(len(detections)))

        if len(tracks) == 0:
            return [], [], list(range(len(detections)))

        # Calculate IoU distance (tracks harus punya .tlbr property)
        dists = iou_distance(tracks, high_conf_detections)

        # ⚠️ PRE-FILTERING: Reject matches dengan IoU terlalu rendah (jelas-jelas salah)
        # Ini membantu reduce false positives
        if self.min_iou_thresh > 0:
            min_cost = 1.0 - self.min_iou_thresh
            dists[dists > min_cost] = np.inf
            logger.debug(f"Stage 1: Pre-filtered with min_iou={self.min_iou_thresh}")

        # Apply score fusion (ByteTrack innovation)
        if self.use_score_fusion:
            dists = fuse_score(dists, high_conf_detections)

        # Run Hungarian algorithm dengan threshold
        matches, unmatched_tracks, unmatched_dets_filtered = linear_assignment(
            dists, thresh=self.stage1_match_thresh
        )

        # Map filtered detection indices back to original indices
        matches_mapped = [
            (track_idx, high_conf_indices[det_idx])
            for track_idx, det_idx in matches
        ]

        # Unmatched detections: low-conf + unmatched high-conf
        low_conf_indices = [i for i in range(len(detections)) if i not in high_conf_indices]
        unmatched_high_conf = [high_conf_indices[i] for i in unmatched_dets_filtered]
        unmatched_detections = sorted(low_conf_indices + unmatched_high_conf)

        logger.debug(f"Stage 1: {len(matches_mapped)} matches, {len(unmatched_tracks)} unmatched tracks, "
                    f"{len(unmatched_detections)} remaining detections")

        return matches_mapped, unmatched_tracks, unmatched_detections

    def match_stage2(self, tracks: List, detections: List[Dict],
                     unmatched_track_indices: List[int], available_det_indices: List[int]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Stage 2: Low-confidence matching (sesuai ByteTrack asli)

        Match remaining TRACKED tracks (bukan lost) dengan low-confidence detections.
        Threshold lebih lenient (0.5) untuk recovery occluded objects.

        Args:
            tracks: List of all Track objects
            detections: List of detection dictionaries
            unmatched_track_indices: Track indices that were not matched in stage 1
            available_det_indices: Detection indices available for matching

        Returns:
            matches: List of (track_idx, det_idx) tuples
            unmatched_tracks: List of still unmatched track indices
            unmatched_detections: List of still unmatched detection indices
        """
        if len(unmatched_track_indices) == 0:
            logger.debug("Stage 2: No unmatched tracks")
            return [], [], available_det_indices

        # Hanya tracks yang statusnya Tracked (bukan Lost)
        # Sesuai ByteTrack asli: r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        unmatched_tracked_tracks = []
        unmatched_tracked_indices = []
        for idx in unmatched_track_indices:
            track = tracks[idx]
            # Check if track is still tracked (not lost/removed)
            if hasattr(track, 'state'):
                if track.state == 'active' or track.state == 'tracked' or getattr(track, 'is_activated', False):
                    unmatched_tracked_tracks.append(track)
                    unmatched_tracked_indices.append(idx)
            else:
                # If no state attribute, assume it's tracked
                unmatched_tracked_tracks.append(track)
                unmatched_tracked_indices.append(idx)

        if len(unmatched_tracked_tracks) == 0:
            logger.debug("Stage 2: No tracked tracks to match")
            return [], unmatched_track_indices, available_det_indices

        # Filter low-confidence detections from available detections
        low_conf_detections = []
        low_conf_indices = []
        for det_idx in available_det_indices:
            conf = detections[det_idx].get('confidence', detections[det_idx].get('score', 0.8))
            if self.stage2_conf_min <= conf < self.stage2_conf_max:
                low_conf_detections.append(detections[det_idx])
                low_conf_indices.append(det_idx)

        if len(low_conf_detections) == 0:
            logger.debug("Stage 2: No low-confidence detections")
            return [], unmatched_track_indices, available_det_indices

        # Calculate IoU distance
        dists = iou_distance(unmatched_tracked_tracks, low_conf_detections)
        
        # ⚠️ PRE-FILTERING: Reject matches dengan IoU terlalu rendah
        if self.min_iou_thresh > 0:
            min_cost = 1.0 - self.min_iou_thresh
            dists[dists > min_cost] = np.inf
            logger.debug(f"Stage 2: Pre-filtered with min_iou={self.min_iou_thresh}")

        # Note: ByteTrack asli TIDAK pakai score fusion untuk stage 2
        # (hanya stage 1 yang pakai fuse_score)

        # Run Hungarian algorithm dengan threshold lebih lenient
        matches, unmatched_tracked_filtered, unmatched_dets_filtered = linear_assignment(
            dists, thresh=self.stage2_match_thresh
        )

        # Map back to original indices
        matches_mapped = [
            (unmatched_tracked_indices[track_idx], low_conf_indices[det_idx])
            for track_idx, det_idx in matches
        ]

        unmatched_tracked_mapped = [unmatched_tracked_indices[i] for i in unmatched_tracked_filtered]
        
        # Unmatched tracks: tracked yang unmatched + tracks yang bukan tracked
        non_tracked_indices = [idx for idx in unmatched_track_indices if idx not in unmatched_tracked_indices]
        unmatched_tracks = unmatched_tracked_mapped + non_tracked_indices

        # Remaining detections: detections not in low_conf + unmatched low_conf
        other_det_indices = [i for i in available_det_indices if i not in low_conf_indices]
        unmatched_low_conf = [low_conf_indices[i] for i in unmatched_dets_filtered]
        unmatched_detections = sorted(other_det_indices + unmatched_low_conf)

        logger.debug(f"Stage 2: {len(matches_mapped)} matches, {len(unmatched_tracks)} unmatched tracks, "
                    f"{len(unmatched_detections)} remaining detections")

        return matches_mapped, unmatched_tracks, unmatched_detections

    def match_unconfirmed(self, unconfirmed_tracks: List, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match unconfirmed tracks (track baru) dengan detections.
        Threshold lebih ketat (0.7) untuk mengurangi ID switches.

        Args:
            unconfirmed_tracks: List of unconfirmed Track objects
            detections: List of detection dictionaries

        Returns:
            matches: List of (track_idx, det_idx) tuples
            unmatched_tracks: List of unmatched track indices
            unmatched_detections: List of unmatched detection indices
        """
        if len(unconfirmed_tracks) == 0 or len(detections) == 0:
            return [], list(range(len(unconfirmed_tracks))), list(range(len(detections)))

        # Calculate IoU distance
        dists = iou_distance(unconfirmed_tracks, detections)
        
        # ⚠️ PRE-FILTERING: Reject matches dengan IoU terlalu rendah
        if self.min_iou_thresh > 0:
            min_cost = 1.0 - self.min_iou_thresh
            dists[dists > min_cost] = np.inf
            logger.debug(f"Unconfirmed: Pre-filtered with min_iou={self.min_iou_thresh}")

        # Apply score fusion untuk unconfirmed juga
        if self.use_score_fusion:
            dists = fuse_score(dists, detections)

        # Run Hungarian dengan threshold lebih ketat untuk track baru
        matches, unmatched_tracks, unmatched_detections = linear_assignment(
            dists, thresh=self.unconfirmed_match_thresh
        )

        logger.debug(f"Unconfirmed matching: {len(matches)} matches, {len(unmatched_tracks)} unmatched tracks")

        return matches, unmatched_tracks, unmatched_detections

    def match_ghost_tracks(self, ghost_tracks: List, detections: List[Dict],
                          available_det_indices: List[int]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Stage 3: Ghost track recovery (TrackGuard novelty!)

        Match ghost tracks with remaining detections after stage 1 and 2.
        Ghost tracks use motion prediction for more lenient matching.

        Args:
            cost_matrix: IoU-based cost matrix (ghost_tracks x detections)
            ghost_tracks: List of ghost Track objects
            detections: List of detection dictionaries
            available_det_indices: Detection indices available for matching

        Returns:
            matches: List of (ghost_track_idx, det_idx) tuples
            unmatched_ghost_tracks: List of unmatched ghost track indices
            unmatched_detections: List of still unmatched detection indices
        """
        if len(ghost_tracks) == 0 or len(available_det_indices) == 0:
            return [], list(range(len(ghost_tracks))), available_det_indices

        # Filter detections untuk yang available
        available_detections = [detections[i] for i in available_det_indices]

        # Calculate IoU distance
        dists = iou_distance(ghost_tracks, available_detections)
        
        # ⚠️ PRE-FILTERING: Reject matches dengan IoU terlalu rendah
        if self.min_iou_thresh > 0:
            min_cost = 1.0 - self.min_iou_thresh
            dists[dists > min_cost] = np.inf
            logger.debug(f"Ghost: Pre-filtered with min_iou={self.min_iou_thresh}")

        # MORE LENIENT threshold for ghost recovery
        ghost_match_thresh = self.ghost_match_thresh  # Use instance variable

        # Run Hungarian algorithm
        matches, unmatched_ghosts, unmatched_dets_filtered = linear_assignment(
            dists, thresh=ghost_match_thresh
        )

        # Map back to original detection indices
        matches_mapped = [
            (ghost_idx, available_det_indices[det_idx])
            for ghost_idx, det_idx in matches
        ]

        unmatched_detections = [available_det_indices[i] for i in unmatched_dets_filtered]

        logger.debug(f"Ghost Recovery: {len(matches_mapped)} ghost tracks recovered, "
                    f"{len(unmatched_ghosts)} ghosts remain, {len(unmatched_detections)} detections unused")

        return matches_mapped, unmatched_ghosts, unmatched_detections
