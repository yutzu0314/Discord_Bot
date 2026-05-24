"""
Matching utilities untuk ByteTrack
===================================

Fungsi-fungsi untuk matching tracks dengan detections:
- IoU distance calculation
- Score fusion (menggabungkan IoU dengan detection confidence)
- Hungarian algorithm wrapper
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

try:
    from cython_bbox import bbox_overlaps as bbox_ious
except ImportError:
    # Fallback jika cython_bbox tidak tersedia
    def bbox_ious(atlbrs, btlbrs):
        """Fallback IoU calculation tanpa cython_bbox"""
        atlbrs = np.ascontiguousarray(atlbrs, dtype=float)
        btlbrs = np.ascontiguousarray(btlbrs, dtype=float)
        
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
        for i, atlbr in enumerate(atlbrs):
            for j, btlbr in enumerate(btlbrs):
                # Calculate IoU
                x1_i = max(atlbr[0], btlbr[0])
                y1_i = max(atlbr[1], btlbr[1])
                x2_i = min(atlbr[2], btlbr[2])
                y2_i = min(atlbr[3], btlbr[3])
                
                if x2_i > x1_i and y2_i > y1_i:
                    intersection = (x2_i - x1_i) * (y2_i - y1_i)
                    area_a = (atlbr[2] - atlbr[0]) * (atlbr[3] - atlbr[1])
                    area_b = (btlbr[2] - btlbr[0]) * (btlbr[3] - btlbr[1])
                    union = area_a + area_b - intersection
                    if union > 0:
                        ious[i, j] = intersection / union
        
        return ious


def ious(atlbrs, btlbrs):
    """
    Compute IoU between bounding boxes.

    Args:
        atlbrs: List of bboxes in format (x1, y1, x2, y2)
        btlbrs: List of bboxes in format (x1, y1, x2, y2)

    Returns:
        np.ndarray: IoU matrix
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(tracks, detections):
    """
    Compute IoU-based distance matrix between tracks and detections.

    Args:
        tracks: List of Track objects with .tlbr property
        detections: List of detection dicts with 'bbox' key (x1, y1, x2, y2)

    Returns:
        np.ndarray: Cost matrix (1 - IoU)
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    atlbrs = [track.tlbr for track in tracks]
    
    # Convert detections to tlbr format
    btlbrs = []
    for det in detections:
        if isinstance(det, dict):
            bbox = det.get('bbox', [])
            if len(bbox) == 4:
                # Assume (x1, y1, x2, y2) or (x, y, w, h)
                if bbox[2] > 1.0 or bbox[3] > 1.0:  # Likely (x1, y1, x2, y2)
                    btlbrs.append(bbox)
                else:  # Likely (x, y, w, h) - convert to (x1, y1, x2, y2)
                    x, y, w, h = bbox
                    btlbrs.append([x, y, x + w, y + h])
            else:
                btlbrs.append([0, 0, 0, 0])
        else:
            # Assume object with bbox attribute
            bbox = getattr(det, 'bbox', [0, 0, 0, 0])
            btlbrs.append(bbox)

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_score(cost_matrix, detections):
    """
    Fuse IoU distance with detection confidence scores.

    This is a key ByteTrack innovation: detection confidence is incorporated
    into the cost matrix to prioritize high-confidence detections.

    Formula: fused_cost = 1 - (iou_similarity * detection_score)

    Args:
        cost_matrix: IoU-based cost matrix (1 - IoU)
        detections: List of detection dicts with 'confidence' key

    Returns:
        np.ndarray: Fused cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1 - cost_matrix  # Convert cost to similarity
    
    # Extract detection scores
    det_scores = []
    for det in detections:
        if isinstance(det, dict):
            score = det.get('confidence', 0.5)
        else:
            score = getattr(det, 'confidence', getattr(det, 'score', 0.5))
        det_scores.append(score)
    
    det_scores = np.array(det_scores, dtype=float)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    
    # Fuse: similarity is weighted by detection score
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim

    return fuse_cost


def linear_assignment(cost_matrix, thresh):
    """
    Solve linear assignment problem using Hungarian algorithm.

    Args:
        cost_matrix: Cost matrix (tracks x detections)
        thresh: Cost threshold - matches with cost > thresh are rejected

    Returns:
        matches: List of (track_idx, det_idx) tuples
        unmatched_tracks: List of unmatched track indices
        unmatched_detections: List of unmatched detection indices
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    # Replace inf with large value for Hungarian
    cost_matrix_clean = np.copy(cost_matrix)
    cost_matrix_clean[np.isinf(cost_matrix_clean)] = 1e9

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix_clean)

    matches = []
    matched_tracks = set()
    matched_dets = set()

    # Filter matches by threshold and valid cost
    for row, col in zip(row_indices, col_indices):
        cost = cost_matrix[row, col]
        if not np.isinf(cost) and cost <= thresh:
            matches.append([row, col])
            matched_tracks.add(row)
            matched_dets.add(col)

    unmatched_tracks = np.where([i not in matched_tracks for i in range(cost_matrix.shape[0])])[0]
    unmatched_detections = np.where([i not in matched_dets for i in range(cost_matrix.shape[1])])[0]

    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_tracks, unmatched_detections

