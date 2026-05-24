"""
Track dengan YOLO11x + ByteTracker + Custom Evaluator
=====================================================

Script untuk evaluasi MOT menggunakan:
1. YOLO11x (best.pt) untuk detection
2. ByteTracker dari repo ByteTrack untuk tracking
3. Custom evaluator (mot_evaluator_hungarian.py) untuk metrik

Usage:
    # Single sequence
    python tools/track_yolo11_with_byte.py \
        --model best.pt \
        --data C:\\Users\\phantom\\TrackGraph-SHA\\isolasi\\MOT17 \
        --sequence MOT17-02-SDP \
        --conf 0.01 \
        --track-thresh 0.6
    
    # List available sequences
    python tools/track_yolo11_with_byte.py \
        --data C:\\Users\\phantom\\TrackGraph-SHA\\isolasi\\MOT17 \
        --list-sequences
    
    # Batch processing multiple sequences
    python tools/track_yolo11_with_byte.py \
        --model best.pt \
        --data C:\\Users\\phantom\\TrackGraph-SHA\\isolasi\\MOT17 \
        --sequences MOT17-02-SDP MOT17-05-SDP MOT17-09-SDP \
        --conf 0.01 \
        --track-thresh 0.6
"""

import argparse
import os
import sys
import cv2
import numpy as np
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path untuk import evaluator
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ByteTrack components
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack

# Import custom components
from tools.yolo11_adapter import YOLO11Adapter
from tools.byte_to_hungarian_converter import ByteTrackToHungarianConverter
from tools.mot17_gt_loader import MOT17GTLoader

# Import evaluator - coba dari mot_evaluator_hungarian.py dulu, fallback ke standalone
try:
    from mot_evaluator_hungarian import SimpleMOTACalculator
    EVALUATOR_AVAILABLE = True
    EVALUATOR_SOURCE = "mot_evaluator_hungarian.py"
except (ImportError, Exception) as e:
    # Fallback ke standalone version
    try:
        from tools.simple_mota_calculator import SimpleMOTACalculator
        EVALUATOR_AVAILABLE = True
        EVALUATOR_SOURCE = "tools/simple_mota_calculator.py (standalone)"
        print(f"Note: Using standalone evaluator (original has dependencies issues)")
    except ImportError:
        print(f"Error: Tidak dapat load evaluator dari kedua sumber!")
        print(f"Pastikan file simple_mota_calculator.py ada di tools/")
        EVALUATOR_AVAILABLE = False
        EVALUATOR_SOURCE = None


class SimpleArgs:
    """Simple args class untuk ByteTracker."""
    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.9, 
                 min_box_area=100, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.mot20 = mot20


def list_available_sequences(data_root: str):
    """
    List semua sequences yang tersedia di dataset.
    
    Args:
        data_root: Root directory MOT17 dataset
    """
    # Cek dengan train/ dulu, kalau tidak ada coba langsung di root
    data_path = Path(data_root) / "train"
    
    if not data_path.exists():
        # Coba langsung di root tanpa folder train
        data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"Error: Path tidak ditemukan: {data_path}")
        return
    
    print("\n" + "=" * 70)
    print("AVAILABLE MOT17 SEQUENCES")
    print("=" * 70)
    
    sequences = []
    for seq_dir in sorted(data_path.iterdir()):
        if seq_dir.is_dir() and seq_dir.name.startswith("MOT17"):
            gt_file = seq_dir / "gt" / "gt.txt"
            img_dir = seq_dir / "img1"
            
            if gt_file.exists() and img_dir.exists():
                # Count images
                img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                sequences.append({
                    'name': seq_dir.name,
                    'images': img_count,
                    'gt_exists': True
                })
    
    if not sequences:
        print("Tidak ada sequence yang ditemukan!")
        return
    
    # Group by detector type
    sdp_seqs = [s for s in sequences if 'SDP' in s['name']]
    frcnn_seqs = [s for s in sequences if 'FRCNN' in s['name']]
    dpm_seqs = [s for s in sequences if 'DPM' in s['name']]
    
    print(f"\n📊 Total: {len(sequences)} sequences ditemukan\n")
    
    if sdp_seqs:
        print("🔵 SDP Sequences:")
        for seq in sdp_seqs:
            seq_num = seq['name'].split('-')[1]
            print(f"   [{seq_num}] {seq['name']:<25} ({seq['images']} frames)")
    
    if frcnn_seqs:
        print("\n🟢 FRCNN Sequences:")
        for seq in frcnn_seqs:
            seq_num = seq['name'].split('-')[1]
            print(f"   [{seq_num}] {seq['name']:<25} ({seq['images']} frames)")
    
    if dpm_seqs:
        print("\n🟡 DPM Sequences:")
        for seq in dpm_seqs:
            seq_num = seq['name'].split('-')[1]
            print(f"   [{seq_num}] {seq['name']:<25} ({seq['images']} frames)")
    
    print("\n" + "=" * 70)
    print("CARA PENGGUNAAN:")
    print("=" * 70)
    print("\n1. Single sequence:")
    print("   python tools/track_yolo11_with_byte.py \\")
    print("       --model best.pt \\")
    print("       --data <path> \\")
    print("       --sequence MOT17-02-SDP")
    print("\n2. Multiple sequences (batch):")
    print("   python tools/track_yolo11_with_byte.py \\")
    print("       --model best.pt \\")
    print("       --data <path> \\")
    print("       --sequences MOT17-02-SDP MOT17-05-SDP MOT17-09-SDP")
    print("\n3. Semua SDP sequences:")
    print("   python tools/track_yolo11_with_byte.py \\")
    print("       --model best.pt \\")
    print("       --data <path> \\")
    print("       --sequences " + " ".join([s['name'] for s in sdp_seqs]))
    print("=" * 70 + "\n")
    
    return [s['name'] for s in sequences]


def make_parser():
    parser = argparse.ArgumentParser("YOLO11x + ByteTracker + Custom Evaluator")
    
    # Model
    parser.add_argument("--model", type=str, default="best.pt",
                       help="Path ke YOLO11x model (.pt file)")
    
    # Dataset
    parser.add_argument("--data", type=str, 
                       default=r"C:\Users\phantom\TrackGraph-SHA\isolasi\MOT17",
                       help="Root directory MOT17 dataset")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Nama sequence untuk evaluasi (contoh: MOT17-02-SDP, MOT17-04-SDP, MOT17-05-SDP). Gunakan --list-sequences untuk lihat semua sequence yang tersedia")
    parser.add_argument("--sequences", type=str, nargs="+", default=None,
                       help="Multiple sequences untuk batch processing (contoh: --sequences MOT17-02-SDP MOT17-05-SDP MOT17-09-SDP). Akan mengevaluasi semua sequence secara berurutan")
    parser.add_argument("--list-sequences", action="store_true",
                       help="List semua sequences yang tersedia di dataset dan exit. Gunakan ini untuk melihat sequence names yang valid")
    
    # Tracking parameters
    parser.add_argument("--track-thresh", type=float, default=0.6,
                       help="Threshold untuk mengaktifkan tracking (default: 0.6, coba 0.3-0.4 untuk lebih banyak tracks)")
    parser.add_argument("--track-buffer", type=int, default=30,
                       help="Buffer untuk menyimpan tracking history (default: 30)")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                       help="Threshold untuk matching detection dengan track (default: 0.8, coba 0.6-0.7 untuk lebih permisif)")
    parser.add_argument("--min-box-area", type=int, default=1,
                       help="Minimum area untuk box (default: 1)")
    parser.add_argument("--mot20", action="store_true",
                       help="Gunakan MOT20 model (default: False)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device untuk menjalankan model (default: cuda)")
    
    # Confidence threshold
    parser.add_argument("--conf", type=float, default=0.01,
                       help="Confidence threshold untuk YOLO11x (default: 0.01)")
    
    return parser


def run_evaluation(args, sequence_name: str):
    """
    Run evaluation untuk satu sequence.
    
    Returns:
        summary: Dictionary dengan hasil evaluasi
    """
    print("\n" + "=" * 70)
    print(f"EVALUATING: {sequence_name}")
    print("=" * 70)
    
    # 1. Load YOLO11x model
    print("\n[1/5] Loading YOLO11x model...")
    try:
        yolo_adapter = YOLO11Adapter(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading YOLO11x model: {e}")
        return None
    
    # 2. Load GT annotations
    print(f"\n[2/5] Loading GT annotations...")
    try:
        gt_loader = MOT17GTLoader(args.data, sequence_name)
        num_frames = gt_loader.get_num_frames()
        frame_ids = gt_loader.get_all_frame_ids()
        print(f"  Total frames: {num_frames}")
        print(f"  Frame range: {min(frame_ids)} - {max(frame_ids)}")
    except Exception as e:
        print(f"Error loading GT: {e}")
        return None
    
    # 3. Initialize ByteTracker
    print(f"\n[3/5] Initializing ByteTracker...")
    tracker_args = SimpleArgs(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        min_box_area=args.min_box_area,
        mot20=args.mot20
    )
    tracker = BYTETracker(tracker_args, frame_rate=30)
    print(f"  Track threshold: {args.track_thresh}")
    print(f"  Match threshold: {args.match_thresh}")
    
    # 4. Initialize Evaluator
    print(f"\n[4/5] Initializing Evaluator...")
    if EVALUATOR_AVAILABLE:
        try:
            evaluator = SimpleMOTACalculator()
            evaluator.reset()
            source = EVALUATOR_SOURCE if 'EVALUATOR_SOURCE' in globals() else "evaluator"
            print(f"  ✓ Evaluator loaded from {source}")
        except Exception as e:
            print(f"  ✗ Error initializing evaluator: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        evaluator = None
        print("  ⚠ Evaluator tidak tersedia")
        print("  Script tidak dapat melanjutkan tanpa evaluator.")
        return None
    
    # 5. Process frames
    print(f"\n[5/5] Processing {num_frames} frames...")
    print("-" * 70)
    
    start_time = time.time()
    converter = ByteTrackToHungarianConverter()
    
    for idx, frame_id in enumerate(frame_ids):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            fps = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {idx+1}/{num_frames} ({fps:.1f} FPS)")
        
        # Load image
        img_path = gt_loader.get_frame_image_path(frame_id)
        if img_path is None:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        img_info = (img_h, img_w, frame_id)
        
        # Run detection dengan YOLO11x
        detections = yolo_adapter.detect(img, conf_threshold=args.conf)
        
        # Convert ke format ByteTrack (tensor)
        # IMPORTANT: ByteTrack expects bboxes in resized format (640x640), not original size
        # YOLO11x returns bboxes in original image size, so we need to scale them
        img_size = (640, 640)
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        
        if len(detections) > 0:
            # Scale bboxes from original size to resized size (640x640)
            detections_scaled = detections.copy()
            detections_scaled[:, :4] = detections[:, :4] * scale
            detections_tensor = torch.from_numpy(detections_scaled).float()
        else:
            detections_tensor = torch.empty((0, 7), dtype=torch.float32)
        
        # Update tracker (ByteTrack will scale back to original size internally)
        online_targets = tracker.update(detections_tensor, img_info, img_size)
        
        # Convert tracking results ke format evaluator
        tracking_results = converter.convert_tracks_to_evaluator_format(
            online_targets, frame_id
        )
        
        # Get GT annotations
        gt_annotations = gt_loader.get_frame_annotations(frame_id)
        
        # Debug info untuk beberapa frame pertama
        if frame_id <= 3:
            print(f"\n  DEBUG Frame {frame_id}:")
            print(f"    Detections: {len(detections)}")
            print(f"    Online targets: {len(online_targets)}")
            print(f"    Tracking results: {len(tracking_results)}")
            print(f"    GT annotations: {len(gt_annotations)}")
            if len(detections) > 0:
                confs = detections[:, 4]  # obj_conf
                print(f"    Detection scores: min={confs.min():.3f}, max={confs.max():.3f}, mean={confs.mean():.3f}")
                print(f"    Scores > 0.6: {np.sum(confs > 0.6)}, Scores 0.3-0.6: {np.sum((confs >= 0.3) & (confs <= 0.6))}, Scores < 0.3: {np.sum(confs < 0.3)}")
                print(f"    ⚠ Detection-to-Track ratio: {len(online_targets)}/{len(detections)} = {len(online_targets)/len(detections)*100:.1f}%")
            if len(tracking_results) > 0:
                print(f"    First track bbox: {tracking_results[0].current_detection['bbox']}")
                print(f"    First track class_id: {tracking_results[0].class_id}")
            print(f"    ⚠ Coverage: {len(online_targets)}/{len(gt_annotations)} GT objects tracked ({len(online_targets)/len(gt_annotations)*100:.1f}%)")
        
        # Update evaluator
        if evaluator is not None:
            evaluator.update_frame(gt_annotations, tracking_results, frame_id)
    
    # Get results
    total_time = time.time() - start_time
    summary = evaluator.get_summary()
    summary['sequence_name'] = sequence_name
    summary['total_time'] = total_time
    summary['fps'] = num_frames / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {sequence_name}")
    print("=" * 70)
    
    print(f"\n📊 FINAL MOTA METRICS:")
    print(f"  MOTA Score: {summary['MOTA']:.3f}")
    print(f"  IDF1 Score: {summary['IDF1']:.3f}")
    print(f"  Precision:  {summary['precision']:.3f}")
    print(f"  Recall:     {summary['recall']:.3f}")
    
    print(f"\n📈 DETAILED ANALYSIS:")
    print(f"  Total GT Objects:    {summary['total_gt']:,}")
    print(f"  False Positives:     {summary['total_fp']:,}")
    print(f"  False Negatives:     {summary['total_fn']:,}")
    print(f"  ID Switches:         {summary['total_id_switches']:,}")
    print(f"  Processed Frames:    {summary['total_frames']:,}")
    
    print(f"\n⏱ Performance:")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average FPS: {summary['fps']:.1f}")
    
    # Save results
    output_file = f"results_yolo11_bytetrack_{sequence_name}.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 70)
    
    return summary


def main():
    args = make_parser().parse_args()
    
    # List sequences mode
    if args.list_sequences:
        list_available_sequences(args.data)
        return
    
    # Validate arguments
    if args.sequence is None and args.sequences is None:
        print("Error: Harus specify --sequence atau --sequences!")
        print("\nGunakan --list-sequences untuk lihat sequences yang tersedia")
        print("Contoh: python tools/track_yolo11_with_byte.py --data <path> --list-sequences")
        return
    
    # Batch processing mode
    if args.sequences:
        print("=" * 70)
        print("BATCH PROCESSING MODE")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.data}")
        print(f"Sequences: {len(args.sequences)}")
        print("=" * 70)
        
        all_results = []
        
        for seq in args.sequences:
            result = run_evaluation(args, seq)
            if result:
                all_results.append(result)
            print("\n")
        
        # Summary semua sequences
        if all_results:
            print("\n" + "=" * 70)
            print("BATCH EVALUATION SUMMARY")
            print("=" * 70)
            
            print(f"\n{'Sequence':<25} {'MOTA':>8} {'IDF1':>8} {'Precision':>10} {'Recall':>8}")
            print("-" * 70)
            
            for result in all_results:
                print(f"{result['sequence_name']:<25} "
                      f"{result['MOTA']:>8.3f} "
                      f"{result['IDF1']:>8.3f} "
                      f"{result['precision']:>10.3f} "
                      f"{result['recall']:>8.3f}")
            
            # Average
            avg_mota = np.mean([r['MOTA'] for r in all_results])
            avg_idf1 = np.mean([r['IDF1'] for r in all_results])
            avg_precision = np.mean([r['precision'] for r in all_results])
            avg_recall = np.mean([r['recall'] for r in all_results])
            
            print("-" * 70)
            print(f"{'AVERAGE':<25} "
                  f"{avg_mota:>8.3f} "
                  f"{avg_idf1:>8.3f} "
                  f"{avg_precision:>10.3f} "
                  f"{avg_recall:>8.3f}")
            
            print("=" * 70)
            
            # Save batch summary
            batch_file = "results_yolo11_bytetrack_batch.json"
            import json
            with open(batch_file, 'w') as f:
                json.dump({
                    'sequences': all_results,
                    'average': {
                        'MOTA': avg_mota,
                        'IDF1': avg_idf1,
                        'precision': avg_precision,
                        'recall': avg_recall
                    }
                }, f, indent=2)
            print(f"\n✓ Batch results saved to: {batch_file}")
    
    # Single sequence mode
    else:
        run_evaluation(args, args.sequence)


if __name__ == "__main__":
    main()
