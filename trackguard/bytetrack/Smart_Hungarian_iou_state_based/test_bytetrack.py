"""
Test script untuk ByteTrack Three-Stage Association

Membandingkan performa:
1. Smart Hungarian (baseline)
2. ByteTrack Three-Stage (High-conf + Low-conf + Ghost)

Sequence: MOT17-02-SDP
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.track_manager_pure import PureSmartHungarianTrackManager
from utils.settings import SETTINGS, enable_bytetrack_association, disable_bytetrack_association
from utils.mot_evaluation import MOTEvaluator

def run_test(use_bytetrack=False, max_frames=100):
    """
    Run tracking test with specified association method

    Args:
        use_bytetrack: If True, use ByteTrack three-stage association
        max_frames: Maximum number of frames to process
    """

    # Configure association method
    if use_bytetrack:
        enable_bytetrack_association()
        method_name = "ByteTrack Three-Stage"
    else:
        disable_bytetrack_association()
        method_name = "Smart Hungarian"

    print(f"\n{'='*70}")
    print(f"Testing: {method_name}")
    print(f"{'='*70}\n")

    # Initialize track manager
    track_manager = PureSmartHungarianTrackManager()

    # Get dataset info
    dataset_info = SETTINGS.get_dataset_info()
    images_path = dataset_info['images_path']
    sequence_name = dataset_info['sequence_name']

    # Get image files
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')])
    print(f"Found {len(image_files)} frames in {sequence_name}")

    # Limit frames for quick test
    if max_frames > 0:
        image_files = image_files[:max_frames]
        print(f"Testing with first {max_frames} frames\n")

    # Process frames
    results = []
    stage_stats = {
        'stage1_total': 0,
        'stage2_total': 0,
        'stage3_total': 0
    }

    for idx, img_file in enumerate(image_files):
        frame_id = idx + 1
        img_path = os.path.join(images_path, img_file)

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Cannot read {img_path}")
            continue

        # Process frame
        pipeline_results = track_manager.process_frame(image, frame_id)

        # Collect stage statistics if using ByteTrack
        if use_bytetrack and 'stage_stats' in pipeline_results['association_results']:
            stats = pipeline_results['association_results']['stage_stats']
            stage_stats['stage1_total'] += stats.get('stage1_matches', 0)
            stage_stats['stage2_total'] += stats.get('stage2_matches', 0)
            stage_stats['stage3_total'] += stats.get('stage3_matches', 0)

        # Get active tracks
        active_tracks = track_manager.get_current_tracks()

        # Store results in MOT format
        for track in active_tracks:
            if track.state == 'active' and track.hits >= 3:
                bbox = track.bbox
                results.append({
                    'frame_id': frame_id,
                    'track_id': track.track_id,
                    'bbox': bbox,
                    'confidence': track.confidence
                })

        # Print progress
        if frame_id % 20 == 0:
            print(f"Frame {frame_id}/{len(image_files)}: {len(active_tracks)} active tracks")

    print(f"\n{method_name} Processing Complete!")
    print(f"Total frames processed: {len(image_files)}")
    print(f"Total track results: {len(results)}")

    if use_bytetrack:
        print(f"\nByteTrack Stage Statistics:")
        print(f"  Stage 1 (High-conf): {stage_stats['stage1_total']} matches")
        print(f"  Stage 2 (Low-conf): {stage_stats['stage2_total']} matches")
        print(f"  Stage 3 (Ghost): {stage_stats['stage3_total']} matches")
        total_matches = sum(stage_stats.values())
        if total_matches > 0:
            print(f"  Stage 1: {100*stage_stats['stage1_total']/total_matches:.1f}%")
            print(f"  Stage 2: {100*stage_stats['stage2_total']/total_matches:.1f}%")
            print(f"  Stage 3: {100*stage_stats['stage3_total']/total_matches:.1f}%")

    return results, track_manager

def evaluate_results(results, method_name):
    """Evaluate tracking results using MOTEvaluator"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*70}\n")

    # Get GT path
    dataset_info = SETTINGS.get_dataset_info()
    gt_path = dataset_info['gt_path']

    if not os.path.exists(gt_path):
        print(f"Warning: Ground truth not found at {gt_path}")
        print("Skipping evaluation")
        return None

    # Initialize evaluator
    evaluator = MOTEvaluator(gt_path)

    # Convert results to MOT format
    mot_results = {}
    for res in results:
        frame_id = res['frame_id']
        if frame_id not in mot_results:
            mot_results[frame_id] = []

        bbox = res['bbox']
        mot_results[frame_id].append({
            'track_id': res['track_id'],
            'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
            'confidence': res['confidence']
        })

    # Evaluate
    metrics = evaluator.evaluate(mot_results)

    # Print metrics
    print(f"\n{method_name} Results:")
    print(f"  MOTA: {metrics.get('mota', 0):.3f}")
    print(f"  IDF1: {metrics.get('idf1', 0):.3f}")
    print(f"  Recall: {metrics.get('recall', 0):.3f}")
    print(f"  Precision: {metrics.get('precision', 0):.3f}")
    print(f"  ID Switches: {metrics.get('num_switches', 0)}")
    print(f"  Fragmentations: {metrics.get('num_fragmentations', 0)}")
    print(f"  False Positives: {metrics.get('num_false_positives', 0)}")
    print(f"  False Negatives: {metrics.get('num_misses', 0)}")

    return metrics

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("ByteTrack Three-Stage Association Test")
    print("="*70)

    # Test parameters
    MAX_FRAMES = 100  # Quick test with first 100 frames

    # Test 1: Smart Hungarian (Baseline)
    print("\n[Test 1/2] Running Smart Hungarian (Baseline)...")
    results_smart, manager_smart = run_test(use_bytetrack=False, max_frames=MAX_FRAMES)

    # Test 2: ByteTrack Three-Stage
    print("\n[Test 2/2] Running ByteTrack Three-Stage...")
    results_bytetrack, manager_bytetrack = run_test(use_bytetrack=True, max_frames=MAX_FRAMES)

    # Evaluate both
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    metrics_smart = evaluate_results(results_smart, "Smart Hungarian")
    metrics_bytetrack = evaluate_results(results_bytetrack, "ByteTrack Three-Stage")

    # Compare
    if metrics_smart and metrics_bytetrack:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)

        mota_diff = metrics_bytetrack['mota'] - metrics_smart['mota']
        idf1_diff = metrics_bytetrack['idf1'] - metrics_smart['idf1']
        idsw_diff = metrics_bytetrack['num_switches'] - metrics_smart['num_switches']

        print(f"\nByteTrack vs Smart Hungarian:")
        print(f"  MOTA: {metrics_bytetrack['mota']:.3f} vs {metrics_smart['mota']:.3f} ({mota_diff:+.3f})")
        print(f"  IDF1: {metrics_bytetrack['idf1']:.3f} vs {metrics_smart['idf1']:.3f} ({idf1_diff:+.3f})")
        print(f"  IDSw: {metrics_bytetrack['num_switches']} vs {metrics_smart['num_switches']} ({idsw_diff:+d})")

        # Verdict
        print(f"\n{'='*70}")
        if idf1_diff > 0.05 and idsw_diff < 0:
            print("✓ ByteTrack Three-Stage shows IMPROVEMENT")
            print(f"  IDF1 improved by {100*idf1_diff:.1f}%")
            print(f"  ID switches reduced by {abs(idsw_diff)}")
        elif idf1_diff > 0:
            print("✓ ByteTrack Three-Stage shows MODERATE improvement")
            print(f"  IDF1 improved by {100*idf1_diff:.1f}%")
        else:
            print("⚠ ByteTrack Three-Stage needs tuning")
            print(f"  IDF1 changed by {100*idf1_diff:.1f}%")
        print(f"{'='*70}\n")

    print("Test complete!")

if __name__ == "__main__":
    main()
