# YOLO11x + ByteTracker Integration

Script untuk evaluasi MOT menggunakan YOLO11x (fine-tuned model Anda) dengan ByteTracker dari repo ByteTrack, dan evaluator custom untuk menghitung metrik MOTA, IDF1, Precision, dan Recall.

## Cara Penggunaan

### 1. Lihat Sequences yang Tersedia

Pertama, lihat dulu sequences apa saja yang ada di dataset Anda:

```bash
python tools/track_yolo11_with_byte.py \
    --data "C:\Users\phantom\TrackGraph-SHA\isolasi\MOT17" \
    --list-sequences
```

Ini akan menampilkan:
- 🔵 SDP Sequences (MOT17-02-SDP, MOT17-05-SDP, dll)
- 🟢 FRCNN Sequences (MOT17-02-FRCNN, dll)
- 🟡 DPM Sequences (MOT17-02-DPM, dll)

### 2. Test Single Sequence

```bash
python tools/track_yolo11_with_byte.py \
    --model best.pt \
    --data "C:\Users\phantom\TrackGraph-SHA\isolasi\MOT17" \
    --sequence MOT17-02-SDP \
    --conf 0.01 \
    --track-thresh 0.6
```

### 3. Test Multiple Sequences (Batch)

```bash
# Test beberapa sequences sekaligus
python tools/track_yolo11_with_byte.py \
    --model best.pt \
    --data "C:\Users\phantom\TrackGraph-SHA\isolasi\MOT17" \
    --sequences MOT17-02-SDP MOT17-05-SDP MOT17-09-SDP \
    --conf 0.01 \
    --track-thresh 0.6

# Test semua SDP sequences
python tools/track_yolo11_with_byte.py \
    --model best.pt \
    --data "C:\Users\phantom\TrackGraph-SHA\isolasi\MOT17" \
    --sequences MOT17-02-SDP MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP \
    --conf 0.01 \
    --track-thresh 0.6
```

## Parameters

### Detection Parameters
- `--model`: Path ke model YOLO11x (.pt file), default: `best.pt`
- `--conf`: Confidence threshold untuk detection, default: `0.01`
- `--nms`: NMS threshold, default: `0.65`

### Tracking Parameters
- `--track-thresh`: Tracking confidence threshold, default: `0.6`
- `--track-buffer`: Buffer size untuk keep lost tracks, default: `30`
- `--match-thresh`: Matching threshold untuk tracking, default: `0.9`
- `--min-box-area`: Minimum box area untuk filtering, default: `100`
- `--mot20`: Use MOT20 settings (flag)

### Dataset Parameters
- `--data`: Root directory MOT17 dataset (wajib)
- `--sequence`: Single sequence name (misal: `MOT17-02-SDP`)
- `--sequences`: Multiple sequences untuk batch (misal: `MOT17-02-SDP MOT17-05-SDP`)
- `--list-sequences`: List semua sequences yang tersedia dan exit

### Other Parameters
- `--device`: Device untuk inference (`cuda` atau `cpu`), default: `cuda`
- `--fp16`: Use FP16 inference (flag)

## Output

### Single Sequence Mode
Script akan menghasilkan:
1. **Progress**: Progress saat processing frames
2. **Metrik Final**:
   - MOTA Score
   - IDF1 Score
   - Precision
   - Recall
3. **Detailed Analysis**:
   - Total GT Objects
   - False Positives
   - False Negatives
   - ID Switches
   - Processed Frames
4. **Results File**: `results_yolo11_bytetrack_<sequence>.json`

### Batch Mode
Script akan menghasilkan:
1. Individual results untuk setiap sequence
2. **Batch Summary Table** dengan rata-rata semua metrics
3. **Batch Results File**: `results_yolo11_bytetrack_batch.json`

## Contoh Output

### Single Sequence
```
======================================================================
EVALUATING: MOT17-02-SDP
======================================================================

[1/5] Loading YOLO11x model...
✓ YOLO11x model loaded from best.pt

[2/5] Loading GT annotations...
✓ Loaded GT: 11844 annotations
  Total frames: 600
  Frame range: 1 - 600

...

======================================================================
RESULTS: MOT17-02-SDP
======================================================================

📊 FINAL MOTA METRICS:
  MOTA Score: 0.698
  IDF1 Score: 0.637
  Precision:  0.869
  Recall:     0.834

📈 DETAILED ANALYSIS:
  Total GT Objects:    11,636
  False Positives:     1,460
  False Negatives:     1,929
  ID Switches:         130
  Processed Frames:    600

⏱ Performance:
  Total time: 125.3s (2.1 min)
  Average FPS: 4.8

✓ Results saved to: results_yolo11_bytetrack_MOT17-02-SDP.json
======================================================================
```

### Batch Mode
```
======================================================================
BATCH EVALUATION SUMMARY
======================================================================

Sequence                  MOTA     IDF1  Precision    Recall
----------------------------------------------------------------------
MOT17-02-SDP            0.698    0.637      0.869    0.834
MOT17-05-SDP            0.712    0.651      0.881    0.845
MOT17-09-SDP            0.685    0.623      0.852    0.821
----------------------------------------------------------------------
AVERAGE                 0.698    0.637      0.867    0.833
======================================================================

✓ Batch results saved to: results_yolo11_bytetrack_batch.json
```

## Troubleshooting

### Error: "ultralytics tidak ditemukan"
```bash
pip install ultralytics
```

### Error: "mot_evaluator_hungarian.py tidak ditemukan"
Pastikan file `mot_evaluator_hungarian.py` ada di root directory (sama level dengan `tools/`).

### Error: "GT file tidak ditemukan"
Pastikan path dataset benar dan struktur folder seperti:
```
MOT17/
  └── train/
      └── MOT17-02-SDP/
          ├── img1/
          │   ├── 000001.jpg
          │   └── ...
          └── gt/
              └── gt.txt
```

### Error: "CUDA out of memory"
- Kurangi `--track-buffer` (misal: 14 atau 25)
- Gunakan `--device cpu` (lebih lambat)
- Test sequence yang lebih pendek dulu

## Catatan

1. Script ini menggunakan ByteTracker dari repo ByteTrack untuk tracking
2. Evaluator menggunakan `SimpleMOTACalculator` dari `mot_evaluator_hungarian.py`
3. Format output evaluator sama dengan yang Anda sudah gunakan sebelumnya
4. Script otomatis convert format YOLO11x → ByteTrack → Evaluator
