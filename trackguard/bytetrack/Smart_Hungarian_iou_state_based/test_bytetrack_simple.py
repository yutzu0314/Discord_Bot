"""
Simple Test Script untuk ByteTrack Three-Stage Association

Menggunakan mot_evaluator.py yang sudah ada untuk evaluasi.
Cukup enable/disable ByteTrack association via settings.

Sequence: MOT17-02-SDP (from settings.py)
"""

import subprocess
import sys
from pathlib import Path

from utils.settings import SETTINGS, enable_bytetrack_association, disable_bytetrack_association

def run_evaluation(use_bytetrack=False):
    """
    Run MOT evaluation with specified association method

    Args:
        use_bytetrack: If True, use ByteTrack three-stage association
    """

    # Configure association method BEFORE importing anything else
    if use_bytetrack:
        print("\n" + "="*70)
        print("RUNNING: ByteTrack Three-Stage Association")
        print("="*70)
        enable_bytetrack_association()
    else:
        print("\n" + "="*70)
        print("RUNNING: Smart Hungarian Association (Baseline)")
        print("="*70)
        disable_bytetrack_association()

    # CRITICAL: Verify setting is actually changed
    print(f"\nVerifying setting: use_bytetrack_association = {SETTINGS.TRACKING_CONFIG['use_bytetrack_association']}")
    if use_bytetrack and not SETTINGS.TRACKING_CONFIG['use_bytetrack_association']:
        print("⚠ WARNING: ByteTrack setting failed to enable!")
        return False
    if not use_bytetrack and SETTINGS.TRACKING_CONFIG['use_bytetrack_association']:
        print("⚠ WARNING: ByteTrack setting failed to disable!")
        return False

    # Print settings
    dataset_info = SETTINGS.get_dataset_info()
    print(f"\nSequence: {dataset_info['sequence_name']}")
    print(f"Frames: {SETTINGS.EXPERIMENT_CONFIG['start_frame']} - {SETTINGS.EXPERIMENT_CONFIG['end_frame']}")

    # Run mot_evaluator.py
    print("\nStarting evaluation...")
    print("-"*70)

    result = subprocess.run(
        [sys.executable, "mot_evaluator.py"],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\n⚠ Evaluation failed with return code {result.returncode}")
        return False

    print("-"*70)
    print("Evaluation complete!\n")
    return True

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("ByteTrack Three-Stage Association Test")
    print("="*70)
    print("\nThis will run evaluation twice:")
    print("1. Smart Hungarian (Baseline)")
    print("2. ByteTrack Three-Stage")
    print("\nResults will be saved in 'results/' directory")
    print("="*70)

    input("\nPress Enter to start Test 1 (Smart Hungarian)...")

    # Test 1: Smart Hungarian (Baseline)
    success1 = run_evaluation(use_bytetrack=False)

    if success1:
        print("\n✓ Test 1 complete! Check results/ for Smart Hungarian metrics")
        print("\nNOTE: Rename/backup result files before running Test 2!")
        input("Press Enter to continue to Test 2 (ByteTrack)...")

        # Test 2: ByteTrack Three-Stage
        success2 = run_evaluation(use_bytetrack=True)

        if success2:
            print("\n✓ Test 2 complete! Check results/ for ByteTrack metrics")
            print("\n" + "="*70)
            print("BOTH TESTS COMPLETE")
            print("="*70)
            print("\nCompare results manually:")
            print("- Smart Hungarian results: results/mot_metrics_*.json (first run)")
            print("- ByteTrack results: results/mot_metrics_*.json (second run)")
            print("\nLook for improvements in:")
            print("  • IDF1 (should increase)")
            print("  • ID Switches (should decrease)")
            print("  • MOTA (may vary slightly)")
            print("="*70)
        else:
            print("\n✗ Test 2 failed")
    else:
        print("\n✗ Test 1 failed")

    print("\nTest script complete!")

if __name__ == "__main__":
    main()
