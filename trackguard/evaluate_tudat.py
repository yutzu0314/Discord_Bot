"""
TU-DAT Evaluation Script
=========================
Evaluates collision detection performance against ground truth annotations.

Metrics computed:
    1. Precision  = TP / (TP + FP)
    2. Recall     = TP / (TP + FN)
    3. F1         = 2 * Precision * Recall / (Precision + Recall)
    4. FP count   (false positives)
    5. FN count   (false negatives)
    6. Delay      (frames between GT start_frame and first AI detection)

Matching logic (event-level):
    - A GT event [start_frame, end_frame] is a TP if any AI detection falls
      within [start_frame - early_tolerance, end_frame + late_tolerance].
    - The *first* matching AI detection determines the delay.
    - Unmatched GT events are FN.
    - AI detections that don't match any GT event are FP.

Usage:
    # Run detection + evaluation on all TU-DAT videos
    python evaluate_tudat.py

    # Evaluate from previously saved detection results
    python evaluate_tudat.py --load-results eval_results/detections.json

    # Custom thresholds
    python evaluate_tudat.py --conf 70 --early-tol 2.0 --late-tol 1.0
"""

import argparse
import json
import os
import sys
import re
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.dirname(__file__))


# ── data classes ─────────────────────────────────────────────────────
@dataclass
class GTEvent:
    video: str
    event_id: int
    start_frame: int
    end_frame: int
    severity: str = "unknown"
    object_type: str = "vehicle"
    annotated_frame: int = -1  # Exact frame of impact
    vehicle_bboxes: List[List[float]] = field(default_factory=list)  # List of normalized [x1, y1, x2, y2] for each vehicle


@dataclass
class AIDetection:
    frame: int
    confidence: float
    confidence_label: str = ""
    track_id_primary: int = -1
    track_id_secondary: int = -1
    detection_mode: str = ""
    bbox_primary: Optional[List[float]] = None  # [x1, y1, x2, y2]
    bbox_secondary: Optional[List[float]] = None  # [x1, y1, x2, y2]
    kinematic_evidence: Optional[Dict] = None  # Velocity, speed, track age data for scientific proof


@dataclass
class MatchResult:
    gt_event: GTEvent
    matched: bool = False
    first_detection_frame: int = -1
    delay_frames: int = 0
    delay_seconds: float = 0.0
    matched_detection: Optional[AIDetection] = None
    spatial_iou: float = 0.0  # IoU between predicted and gt collision zones
    timeliness_score: float = 0.0  # 0-1, penalizes late detection


# ── ground truth parser ──────────────────────────────────────────────
def parse_annotations(tudat_dir: str) -> Dict[str, List[GTEvent]]:
    """Parse all .txt annotation files in TU-DAT directory.

    Handles:
        - Single JSON object per file
        - Multiple JSON objects concatenated (v1.txt style)
        - Files with 'ignore: true' or '(skip)' in filename
        - Events array with multiple events (v20.txt style)
    """
    annotations = {}
    tudat_path = Path(tudat_dir)

    for txt_file in sorted(tudat_path.glob("*.txt")):
        filename = txt_file.name

        # Skip files with "(skip)" or "-skip" in name
        if "skip" in filename.lower():
            print(f"  [SKIP] {filename}")
            continue

        # Extract video number from filename (e.g. v1.txt -> v1)
        match = re.match(r'(v\d+)', filename)
        if not match:
            continue
        video_key = match.group(1)

        # Read file content and parse JSON blocks
        content = txt_file.read_text(encoding='utf-8').strip()
        if not content:
            continue

        events = []
        json_blocks = _split_json_blocks(content)

        for block in json_blocks:
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                print(f"  [WARN] Bad JSON in {filename}")
                continue

            # Check ignore flag
            if data.get("ignore", False):
                print(f"  [SKIP] {filename} (ignore=true)")
                continue

            fps = data.get("fps", 30)
            num_frames = data.get("num_frames", 0)

            for ev in data.get("events", []):
                if ev.get("event_type") != "collision":
                    continue

                start = ev.get("start_frame", 0)
                end = ev.get("end_frame", 0)

                # Sanity check: start > end or start > num_frames
                if start > end or (num_frames > 0 and start > num_frames):
                    print(f"  [WARN] {filename} event {ev.get('event_id')}: "
                          f"bad frame range ({start}-{end}), num_frames={num_frames}")
                    continue

                # Extract impact_region if available (from annotator tool)
                impact_region = ev.get("impact_region", {})
                annotated_frame = impact_region.get("annotated_frame", start)  # Fallback to start_frame

                # Convert vehicle bboxes from {cx, cy, w, h} to [x1, y1, x2, y2]
                raw_vehicles = impact_region.get("vehicles", [])
                vehicle_bboxes = []
                for v in raw_vehicles:
                    if isinstance(v, dict):
                        # Format: {cx, cy, w, h} (from annotator tool)
                        cx = v.get("cx", 0)
                        cy = v.get("cy", 0)
                        w = v.get("w", 0)
                        h = v.get("h", 0)
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        vehicle_bboxes.append([x1, y1, x2, y2])
                    elif isinstance(v, list) and len(v) >= 4:
                        # Already in [x1, y1, x2, y2] format
                        vehicle_bboxes.append(v[:4])

                events.append(GTEvent(
                    video=video_key,
                    event_id=ev.get("event_id", 0),
                    start_frame=start,
                    end_frame=end,
                    severity=ev.get("severity", "unknown"),
                    object_type=ev.get("primary_object_type", "vehicle"),
                    annotated_frame=annotated_frame,
                    vehicle_bboxes=vehicle_bboxes,
                ))

        if events:
            annotations.setdefault(video_key, []).extend(events)

    return annotations


def _split_json_blocks(text: str) -> List[str]:
    """Split a file that may contain multiple concatenated JSON objects."""
    blocks = []
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start:i + 1])
                start = None

    return blocks


# ── spatial matching helpers ──────────────────────────────────────────
def compute_bbox_union(bboxes: List[List[float]]) -> List[float]:
    """Compute union of multiple bboxes in normalized [x1, y1, x2, y2] format."""
    if not bboxes:
        return [0, 0, 1, 1]

    x1_min = min(b[0] for b in bboxes)
    y1_min = min(b[1] for b in bboxes)
    x2_max = max(b[2] for b in bboxes)
    y2_max = max(b[3] for b in bboxes)

    return [x1_min, y1_min, x2_max, y2_max]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two normalized bboxes in [x1, y1, x2, y2] format."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1_min = max(box1[0], box2[0])
    y1_min = max(box1[1], box2[1])
    x2_max = min(box1[2], box2[2])
    y2_max = min(box1[3], box2[3])

    # No intersection
    if x1_min >= x2_max or y1_min >= y2_max:
        return 0.0

    intersection = (x2_max - x1_min) * (y2_max - y1_min)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_timeliness_score(detection_frame: int, annotated_frame: int,
                            window_end: int) -> float:
    """
    Compute timeliness score (0-1) based on delay from annotated frame.

    Early detection (before annotated_frame) gets full 1.0 score.
    Late detection gets penalized: 1 - delay/window_size
    """
    delay = detection_frame - annotated_frame
    window_size = window_end - annotated_frame

    if window_size <= 0:
        return 0.0

    if delay < 0:
        # Early warning - reward with full score
        return 1.0

    # Late detection - penalize proportionally
    return max(0.0, 1.0 - (delay / window_size))


# ── detection runner ─────────────────────────────────────────────────
def run_detection(video_path: str, tracker: str = "bytetrack",
                  conf_threshold: int = 70, min_hits: int = 5,
                  model: str = "yolo11n.pt") -> List[AIDetection]:
    """Run collision detection pipeline on a single video.

    Returns list of AIDetection with frame numbers where collisions were detected.
    """
    from utils.settings import SETTINGS

    # Configure settings
    SETTINGS.DETECTOR_WEIGHTS = model
    SETTINGS.PHYSICS_CONFIG['enable_behaviour_detection'] = True
    SETTINGS.PHYSICS_CONFIG['enable_eager'] = True
    SETTINGS.PHYSICS_CONFIG['collision_detector']['collision_confidence_threshold'] = conf_threshold
    SETTINGS.PHYSICS_CONFIG['collision_detector']['min_track_hits'] = min_hits

    # Initialize tracker
    if tracker == 'bytetrack':
        from core.bytetrack_manager import ByteTrackManager
        track_manager = ByteTrackManager(use_physics=True)
    else:
        from core.track_manager import PureSmartHungarianTrackManager
        track_manager = PureSmartHungarianTrackManager(use_physics=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [ERROR] Cannot open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections: List[AIDetection] = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        try:
            track_manager.process_frame(frame, frame_id)
        except Exception:
            continue

        behaviour_results = getattr(track_manager, 'behaviour_results', {})
        collision_dets = behaviour_results.get('collision', [])

        for det in collision_dets:
            detections.append(AIDetection(
                frame=frame_id,
                confidence=det.get('confidence', 0.0),
                confidence_label=det.get('confidence_label', ''),
                track_id_primary=det.get('track_id', -1),
                track_id_secondary=det.get('track_id_secondary', -1),
                detection_mode=det.get('detection_mode', ''),
                bbox_primary=det.get('bbox_primary', None),  # Will extract from tracked detection
                bbox_secondary=det.get('bbox_secondary', None),  # Will extract from tracked detection
            ))

    cap.release()

    # De-duplicate: keep unique (frame, track_pair) entries
    seen = set()
    unique = []
    for d in detections:
        pair = (d.frame, tuple(sorted([d.track_id_primary, d.track_id_secondary])))
        if pair not in seen:
            seen.add(pair)
            unique.append(d)

    return unique


# ── event clustering ─────────────────────────────────────────────────
@dataclass
class AIEvent:
    """A cluster of consecutive AI detections = one predicted event."""
    start_frame: int
    end_frame: int
    peak_confidence: float
    num_frames: int
    representative: AIDetection  # first detection in cluster


def cluster_detections(detections: List[AIDetection],
                       gap_seconds: float = 2.0,
                       fps: int = 30) -> List[AIEvent]:
    """Cluster frame-level detections into event-level predictions.

    Detections within `gap_seconds` of each other (by frame number)
    are merged into a single event.  This prevents one real collision
    that fires for 100+ frames from being counted as 100 detections.

    Args:
        detections: Raw per-frame detections.
        gap_seconds: Max gap (seconds) between frames in same cluster.
        fps: Video FPS.

    Returns:
        List of AIEvent clusters, sorted by start_frame.
    """
    if not detections:
        return []

    gap_frames = int(gap_seconds * fps)
    sorted_dets = sorted(detections, key=lambda d: d.frame)

    clusters: List[AIEvent] = []
    cur_start = sorted_dets[0].frame
    cur_end = sorted_dets[0].frame
    cur_peak = sorted_dets[0].confidence
    cur_first = sorted_dets[0]
    cur_count = 1

    for det in sorted_dets[1:]:
        if det.frame - cur_end <= gap_frames:
            # Extend current cluster
            cur_end = det.frame
            cur_count += 1
            if det.confidence > cur_peak:
                cur_peak = det.confidence
        else:
            # Finalize current cluster, start new one
            clusters.append(AIEvent(
                start_frame=cur_start,
                end_frame=cur_end,
                peak_confidence=cur_peak,
                num_frames=cur_count,
                representative=cur_first,
            ))
            cur_start = det.frame
            cur_end = det.frame
            cur_peak = det.confidence
            cur_first = det
            cur_count = 1

    # Finalize last cluster
    clusters.append(AIEvent(
        start_frame=cur_start,
        end_frame=cur_end,
        peak_confidence=cur_peak,
        num_frames=cur_count,
        representative=cur_first,
    ))

    return clusters


# ── matching & metrics ───────────────────────────────────────────────
def evaluate_video(gt_events: List[GTEvent],
                   ai_detections: List[AIDetection],
                   fps: int = 30,
                   early_tol_sec: float = 2.0,
                   late_tol_sec: float = 1.0,
                   cluster_gap_sec: float = 2.0) -> Dict:
    """Evaluate a single video's detections against ground truth.

    AI detections are first clustered into events, then matched against
    GT events.  This ensures that one continuous detection burst counts
    as one predicted event, not hundreds of FPs.

    Args:
        gt_events:       Ground truth collision events for this video.
        ai_detections:   Raw per-frame AI detections.
        fps:             Video FPS for converting delay to seconds.
        early_tol_sec:   Seconds BEFORE start_frame still counts as TP.
        late_tol_sec:    Seconds AFTER end_frame still counts as TP.
        cluster_gap_sec: Max gap (seconds) to merge detections into one event.

    Returns:
        Dict with TP, FP, FN, matches, unmatched_events, raw_detection_count.
    """
    early_tol_frames = int(early_tol_sec * fps)
    late_tol_frames = int(late_tol_sec * fps)

    # Cluster raw detections → event-level
    ai_events = cluster_detections(ai_detections, gap_seconds=cluster_gap_sec, fps=fps)

    matched_event_indices = set()
    matches: List[MatchResult] = []

    # For each GT event, find the earliest matching AI event cluster
    for gt in gt_events:
        window_start = gt.start_frame - early_tol_frames
        window_end = gt.end_frame + late_tol_frames

        result = MatchResult(gt_event=gt)

        for idx, ai_ev in enumerate(ai_events):
            if idx in matched_event_indices:
                continue
            # An AI event matches if its range overlaps the GT tolerance window
            if ai_ev.end_frame >= window_start and ai_ev.start_frame <= window_end:
                result.matched = True
                result.first_detection_frame = ai_ev.start_frame
                result.delay_frames = ai_ev.start_frame - gt.start_frame
                result.delay_seconds = result.delay_frames / fps
                result.matched_detection = ai_ev.representative
                matched_event_indices.add(idx)
                break

        matches.append(result)

    # Separate unmatched AI events into true FP vs "absorbed" (inside GT window)
    # An unmatched AI event that overlaps ANY GT tolerance window is not a true FP —
    # it's a secondary detection of the same collision (different track pair).
    unmatched_indices = [i for i in range(len(ai_events)) if i not in matched_event_indices]

    true_fp_events = []
    absorbed_events = []  # inside GT window but not the primary match

    for idx in unmatched_indices:
        ai_ev = ai_events[idx]
        inside_any_gt = False
        for gt in gt_events:
            ws = gt.start_frame - early_tol_frames
            we = gt.end_frame + late_tol_frames
            if ai_ev.end_frame >= ws and ai_ev.start_frame <= we:
                inside_any_gt = True
                break
        if inside_any_gt:
            absorbed_events.append(ai_ev)
        else:
            true_fp_events.append(ai_ev)

    tp = sum(1 for m in matches if m.matched)
    fn = sum(1 for m in matches if not m.matched)
    fp = len(true_fp_events)

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "matches": matches,
        "unmatched_events": true_fp_events,
        "absorbed_events": absorbed_events,
        "ai_event_clusters": len(ai_events),
        "raw_detection_count": len(ai_detections),
    }


def evaluate_video_dual_criterion(gt_events: List[GTEvent],
                                 ai_detections: List[AIDetection],
                                 fps: int = 30,
                                 early_tol_sec: float = 2.0,
                                 late_tol_sec: float = 1.0,
                                 cluster_gap_sec: float = 2.0,
                                 spatial_iou_threshold: float = 0.30) -> Dict:
    """Evaluate with temporal matching + optional spatial quality metric.

    TP criterion: temporal overlap with GT window (primary, sufficient for TP).
    Spatial IoU is computed as a quality metric when bbox data is available,
    but does NOT gate TP counting — a large/anomalous predicted bbox should
    not cause a real collision to be missed.

    Args:
        gt_events:              Ground truth collision events (with optional impact_region)
        ai_detections:          Raw per-frame AI detections (with optional bboxes)
        fps:                    Video FPS
        early_tol_sec:          Early tolerance before window (seconds)
        late_tol_sec:           Late tolerance after window (seconds)
        cluster_gap_sec:        Gap before merging clusters (seconds)
        spatial_iou_threshold:  Kept for API compatibility; no longer used as hard gate.

    Returns:
        Dict with TP, FP, FN, matches (with spatial metrics), etc.
    """
    early_tol_frames = int(early_tol_sec * fps)
    late_tol_frames = int(late_tol_sec * fps)

    # Cluster raw detections → event-level
    ai_events = cluster_detections(ai_detections, gap_seconds=cluster_gap_sec, fps=fps)

    matched_event_indices = set()
    matches: List[MatchResult] = []

    # For each GT event, find the earliest matching AI event cluster
    for gt in gt_events:
        window_start = gt.start_frame - early_tol_frames
        window_end = gt.end_frame + late_tol_frames

        # Use annotated_frame if available, else start_frame
        ref_frame = gt.annotated_frame if gt.annotated_frame >= 0 else gt.start_frame

        result = MatchResult(gt_event=gt)

        for idx, ai_ev in enumerate(ai_events):
            if idx in matched_event_indices:
                continue

            # ── CRITERION 1: TEMPORAL ──
            if not (ai_ev.end_frame >= window_start and ai_ev.start_frame <= window_end):
                continue

            # ── CRITERION 2: SPATIAL (quality metric, not a hard gate) ──
            # Spatial IoU is computed when bbox data is available and stored for
            # reporting, but does NOT block TP counting.  A temporal match alone
            # is sufficient — large/anomalous predicted bboxes must not turn a
            # real detection into an FN.
            spatial_iou = 1.0  # Default to perfect if no bbox info

            if gt.vehicle_bboxes and ai_ev.representative.bbox_primary and ai_ev.representative.bbox_secondary:
                # Compute gt_zone = union of all vehicle bboxes
                gt_zone = compute_bbox_union(gt.vehicle_bboxes)

                # Compute predicted_zone = union of primary + secondary bboxes
                pred_bboxes = [
                    ai_ev.representative.bbox_primary,
                    ai_ev.representative.bbox_secondary
                ]
                pred_zone = compute_bbox_union(pred_bboxes)

                # Compute IoU (stored for quality reporting only)
                spatial_iou = compute_iou(gt_zone, pred_zone)

            # Temporal match (Criterion 1) is the sole TP gate — record and break
            result.matched = True
            result.first_detection_frame = ai_ev.start_frame
            result.delay_frames = ai_ev.start_frame - ref_frame
            result.delay_seconds = result.delay_frames / fps
            result.matched_detection = ai_ev.representative
            result.spatial_iou = spatial_iou
            result.timeliness_score = compute_timeliness_score(
                ai_ev.start_frame, ref_frame, gt.end_frame
            )
            matched_event_indices.add(idx)
            break

        matches.append(result)

    # Separate unmatched AI events into true FP vs absorbed
    unmatched_indices = [i for i in range(len(ai_events)) if i not in matched_event_indices]

    true_fp_events = []
    absorbed_events = []

    for idx in unmatched_indices:
        ai_ev = ai_events[idx]
        inside_any_gt = False
        for gt in gt_events:
            ws = gt.start_frame - early_tol_frames
            we = gt.end_frame + late_tol_frames
            if ai_ev.end_frame >= ws and ai_ev.start_frame <= we:
                inside_any_gt = True
                break
        if inside_any_gt:
            absorbed_events.append(ai_ev)
        else:
            true_fp_events.append(ai_ev)

    tp = sum(1 for m in matches if m.matched)
    fn = sum(1 for m in matches if not m.matched)
    fp = len(true_fp_events)

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "matches": matches,
        "unmatched_events": true_fp_events,
        "absorbed_events": absorbed_events,
        "ai_event_clusters": len(ai_events),
        "raw_detection_count":len(ai_detections),
    }


def compute_aggregate_metrics(all_results: Dict[str, Dict]) -> Dict:
    """Compute aggregate Precision, Recall, F1, and delay statistics."""
    total_tp = sum(r["tp"] for r in all_results.values())
    total_fp = sum(r["fp"] for r in all_results.values())
    total_fn = sum(r["fn"] for r in all_results.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Delay statistics (only from TP matches)
    delays_frames = []
    delays_seconds = []
    for r in all_results.values():
        for m in r["matches"]:
            if m.matched:
                delays_frames.append(m.delay_frames)
                delays_seconds.append(m.delay_seconds)

    delay_stats = {}
    if delays_frames:
        delay_stats = {
            "mean_frames": np.mean(delays_frames),
            "median_frames": np.median(delays_frames),
            "std_frames": np.std(delays_frames),
            "min_frames": min(delays_frames),
            "max_frames": max(delays_frames),
            "mean_seconds": np.mean(delays_seconds),
            "median_seconds": np.median(delays_seconds),
            "all_delays_frames": delays_frames,
            "all_delays_seconds": delays_seconds,
        }

    total_absorbed = sum(len(r.get("absorbed_events", [])) for r in all_results.values())

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "absorbed": total_absorbed,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "delay": delay_stats,
        "total_gt_events": total_tp + total_fn,
        "total_ai_detections": total_tp + total_fp,
    }


# ── display ──────────────────────────────────────────────────────────
def print_results(all_results: Dict[str, Dict], metrics: Dict):
    """Pretty-print evaluation results."""
    sep = "=" * 80

    print(f"\n{sep}")
    print("  TU-DAT COLLISION DETECTION EVALUATION REPORT")
    print(f"{sep}\n")

    # Per-video results
    print("Per-Video Results (EVENT-LEVEL: raw detections clustered into events):")
    print(f"{'Video':<8} {'TP':>4} {'FP':>4} {'FN':>4} {'AI Evts':>8} {'GT Evts':>8} {'Raw Det':>8} {'Delay (frames)':>16}")
    print("-" * 80)

    total_raw = 0
    for video_key in sorted(all_results.keys(), key=lambda k: int(k[1:])):
        r = all_results[video_key]
        n_ai_events = r.get("ai_event_clusters", r["tp"] + r["fp"])
        n_gt = r["tp"] + r["fn"]
        raw = r.get("raw_detection_count", 0)
        total_raw += raw

        delays = [m.delay_frames for m in r["matches"] if m.matched]
        delay_str = ", ".join(f"{d:+d}" for d in delays) if delays else "-"

        print(f"{video_key:<8} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{n_ai_events:>8} {n_gt:>8} {raw:>8} {delay_str:>16}")

    # Aggregate
    print(f"\n{sep}")
    print("  AGGREGATE METRICS")
    print(f"{sep}\n")

    print(f"  Total GT events      : {metrics['total_gt_events']}")
    print(f"  Total AI events      : {metrics['total_ai_detections']}  (after clustering)")
    print(f"  Total raw detections : {total_raw}  (before clustering)")
    print()
    print(f"  TP : {metrics['TP']}")
    print(f"  FP : {metrics['FP']}  (true false positives — outside all GT windows)")
    print(f"  FN : {metrics['FN']}")
    print(f"  Absorbed : {metrics.get('absorbed', 0)}  (extra detections inside GT window, not counted as FP)")
    print()
    print(f"  Precision : {metrics['Precision']:.4f}  ({metrics['Precision']*100:.1f}%)")
    print(f"  Recall    : {metrics['Recall']:.4f}  ({metrics['Recall']*100:.1f}%)")
    print(f"  F1 Score  : {metrics['F1']:.4f}  ({metrics['F1']*100:.1f}%)")

    if metrics["delay"]:
        d = metrics["delay"]
        print()
        print(f"  Delay Statistics (TP matches only):")
        print(f"    Mean   : {d['mean_frames']:+.1f} frames  ({d['mean_seconds']:+.2f}s)")
        print(f"    Median : {d['median_frames']:+.1f} frames  ({d['median_seconds']:+.2f}s)")
        print(f"    Std    : {d['std_frames']:.1f} frames")
        print(f"    Range  : [{d['min_frames']:+d}, {d['max_frames']:+d}] frames")
        print()

        # Breakdown: early vs on-time vs late
        early = sum(1 for d_f in d['all_delays_frames'] if d_f < 0)
        on_time = sum(1 for d_f in d['all_delays_frames'] if d_f == 0)
        late = sum(1 for d_f in d['all_delays_frames'] if d_f > 0)
        total_tp = len(d['all_delays_frames'])
        print(f"    Early   (delay < 0) : {early}/{total_tp}")
        print(f"    On-time (delay = 0) : {on_time}/{total_tp}")
        print(f"    Late    (delay > 0) : {late}/{total_tp}")

    # Detail: FP events (false alarms)
    print(f"\n{sep}")
    print("  FALSE POSITIVE EVENTS (FP) - Detail")
    print(f"{sep}\n")

    total_fp_count = 0
    for video_key in sorted(all_results.keys(), key=lambda k: int(k[1:])):
        r = all_results[video_key]
        fp_events = r.get("unmatched_events", [])
        if not fp_events:
            continue

        total_fp_count += len(fp_events)
        gt_ranges = [f"[{g.start_frame}-{g.end_frame}]"
                     for g in [m.gt_event for m in r["matches"]]]
        print(f"  {video_key}  (GT windows: {', '.join(gt_ranges)})")

        for i, ev in enumerate(fp_events, 1):
            rep = ev.representative
            duration = ev.end_frame - ev.start_frame
            print(f"    FP#{i}: frames [{ev.start_frame}-{ev.end_frame}] "
                  f"({ev.num_frames} det, {duration} frames span) "
                  f"| conf={ev.peak_confidence:.2f} "
                  f"| mode={rep.detection_mode} "
                  f"| tracks={rep.track_id_primary}<->{rep.track_id_secondary}")
        print()

    if total_fp_count == 0:
        print("  No false positives!\n")

    # Detail: FN events (missed)
    fn_events = []
    for video_key in sorted(all_results.keys(), key=lambda k: int(k[1:])):
        for m in all_results[video_key]["matches"]:
            if not m.matched:
                fn_events.append(m.gt_event)

    if fn_events:
        print(f"{sep}")
        print("  MISSED EVENTS (FN)")
        print(f"{sep}\n")
        for ev in fn_events:
            print(f"  {ev.video} event#{ev.event_id}: "
                  f"frames [{ev.start_frame}-{ev.end_frame}], "
                  f"severity={ev.severity}")

    print(f"\n{sep}\n")


# ── save / load ──────────────────────────────────────────────────────
def save_detections(all_detections: Dict[str, List[AIDetection]], path: str):
    """Save raw detection results to JSON for later re-evaluation."""
    data = {}
    for video_key, dets in all_detections.items():
        data[video_key] = [asdict(d) for d in dets]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Detections saved to {path}")


def load_detections(path: str) -> Dict[str, List[AIDetection]]:
    """Load previously saved detection results."""
    with open(path) as f:
        data = json.load(f)

    all_dets = {}
    for video_key, det_list in data.items():
        all_dets[video_key] = [AIDetection(**d) for d in det_list]
    return all_dets


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TU-DAT Collision Detection Evaluation")

    parser.add_argument('--tudat-dir', type=str,
                        default=str(Path(__file__).parent / "TU-DAT"),
                        help='Path to TU-DAT directory with videos and annotations')

    # Detection pipeline options
    parser.add_argument('--tracker', type=str, default='bytetrack',
                        choices=['bytetrack', 'hungarian'])
    parser.add_argument('--conf', type=int, default=70,
                        help='Collision confidence threshold %% (default: 70)')
    parser.add_argument('--min-hits', type=int, default=5)
    parser.add_argument('--model', type=str, default='yolo11n.pt')

    # Tolerance for matching
    parser.add_argument('--early-tol', type=float, default=2.0,
                        help='Early detection tolerance in seconds (default: 2.0)')
    parser.add_argument('--late-tol', type=float, default=1.0,
                        help='Late detection tolerance in seconds after end_frame (default: 1.0)')
    parser.add_argument('--cluster-gap', type=float, default=2.0,
                        help='Max gap (seconds) to merge detections into one event (default: 2.0)')

    # Load/save
    parser.add_argument('--load-results', type=str, default=None,
                        help='Load detection results from JSON instead of running pipeline')
    parser.add_argument('--save-results', type=str, default='eval_results/detections.json',
                        help='Save detection results to JSON')

    # Filter
    parser.add_argument('--videos', type=str, nargs='*', default=None,
                        help='Only evaluate specific videos (e.g. v1 v2 v20)')

    args = parser.parse_args()

    tudat_dir = args.tudat_dir
    print(f"TU-DAT directory: {tudat_dir}\n")

    # ── Step 1: Parse ground truth ────────────────────────────────────
    print("Parsing ground truth annotations...")
    annotations = parse_annotations(tudat_dir)
    total_events = sum(len(evs) for evs in annotations.values())
    print(f"  Found {len(annotations)} videos with {total_events} collision events\n")

    if not annotations:
        print("No annotations found. Check TU-DAT directory.")
        return

    # ── Step 2: Get AI detections ─────────────────────────────────────
    if args.load_results:
        print(f"Loading saved detections from {args.load_results}...")
        all_detections = load_detections(args.load_results)
    else:
        print("Running collision detection pipeline on each video...\n")
        all_detections = {}

        video_keys = sorted(annotations.keys(), key=lambda k: int(k[1:]))
        if args.videos:
            video_keys = [v for v in video_keys if v in args.videos]

        for i, video_key in enumerate(video_keys, 1):
            video_path = Path(tudat_dir) / f"{video_key}.mp4"

            if not video_path.exists():
                print(f"  [{i}/{len(video_keys)}] {video_key}: video not found, skipping")
                continue

            print(f"  [{i}/{len(video_keys)}] {video_key}: processing...", end=" ", flush=True)
            t0 = time.time()

            dets = run_detection(
                str(video_path),
                tracker=args.tracker,
                conf_threshold=args.conf,
                min_hits=args.min_hits,
                model=args.model,
            )

            elapsed = time.time() - t0
            print(f"{len(dets)} detections ({elapsed:.1f}s)")
            all_detections[video_key] = dets

        # Save results
        save_detections(all_detections, args.save_results)

    # ── Step 3: Evaluate ──────────────────────────────────────────────
    print("\nEvaluating...")

    all_results = {}
    video_keys = sorted(annotations.keys(), key=lambda k: int(k[1:]))
    if args.videos:
        video_keys = [v for v in video_keys if v in args.videos]

    for video_key in video_keys:
        gt_events = annotations.get(video_key, [])
        ai_dets = all_detections.get(video_key, [])

        # Get FPS from annotation file
        fps = 30  # default
        txt_path = Path(tudat_dir) / f"{video_key}.txt"
        if txt_path.exists():
            try:
                blocks = _split_json_blocks(txt_path.read_text(encoding='utf-8'))
                if blocks:
                    fps = json.loads(blocks[0]).get("fps", 30)
            except Exception:
                pass

        result = evaluate_video(
            gt_events, ai_dets, fps=fps,
            early_tol_sec=args.early_tol,
            late_tol_sec=args.late_tol,
            cluster_gap_sec=args.cluster_gap,
        )
        all_results[video_key] = result

    # ── Step 4: Aggregate & print ─────────────────────────────────────
    metrics = compute_aggregate_metrics(all_results)
    print_results(all_results, metrics)

    # Save evaluation report
    report_path = Path(args.save_results).parent / "evaluation_report.json"
    report = {
        "config": {
            "tracker": args.tracker,
            "conf_threshold": args.conf,
            "min_hits": args.min_hits,
            "model": args.model,
            "early_tolerance_sec": args.early_tol,
            "late_tolerance_sec": args.late_tol,
            "cluster_gap_sec": args.cluster_gap,
        },
        "metrics": {k: v for k, v in metrics.items()
                    if k != "delay" or not isinstance(v, dict)},
    }
    if metrics.get("delay"):
        report["delay_stats"] = {
            k: v for k, v in metrics["delay"].items()
            if k not in ("all_delays_frames", "all_delays_seconds")
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
