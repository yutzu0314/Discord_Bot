# Threshold Tuning untuk Mengurangi FP dan ID Switches

## Masalah yang Teridentifikasi

Dari hasil evaluasi:
- **ID Switches: 156** (vs 64-75 sebelumnya) ❌ **EXPLODED**
- **FP: 5725** (vs 339-633 sebelumnya) ❌ **SANGAT TINGGI**
- **FN: 12872** (vs 4236-12731 sebelumnya) ⚠️ Tinggi
- **MOTA: 0.604** (vs 0.647 sebelumnya) ⚠️ Turun

## Analisis Root Cause

1. **Threshold terlalu longgar** - Terlalu banyak false matches diterima
2. **Tidak ada pre-filtering IoU** - Matches dengan IoU sangat rendah (< 0.3) masih masuk Hungarian
3. **Stage 2 terlalu agresif** - Threshold 0.5 terlalu longgar untuk recovery

## Perbaikan yang Diterapkan

### 1. Threshold Dinaikkan

**Sebelum:**
- Stage 2: `0.5` 
- Unconfirmed: `0.7`
- Ghost: `0.4`

**Sesudah:**
- Stage 2: `0.6` ⬆️ (lebih ketat untuk reduce FP)
- Unconfirmed: `0.75` ⬆️ (lebih ketat untuk track baru)
- Ghost: `0.55` ⬆️ (lebih ketat untuk reduce ID switches)

### 2. Pre-filtering IoU Ditambahkan

**Minimum IoU threshold: `0.3`**

Sebelum Hungarian algorithm, kita reject matches dengan IoU < 0.3:
- Ini mengurangi false positives yang jelas-jelas salah
- Matches dengan overlap sangat kecil tidak akan dipertimbangkan
- Aplikasi di semua stage: Stage 1, Stage 2, Unconfirmed, Ghost

### 3. Logging Improved

Threshold values sekarang di-log untuk monitoring.

## Expected Impact

Dengan threshold lebih ketat dan pre-filtering:
- ✅ **ID Switches turun** - Lebih sedikit false associations
- ✅ **FP turun** - Lebih sedikit false detections diterima
- ⚠️ **FN mungkin naik sedikit** - Tapi trade-off yang wajar untuk accuracy
- ✅ **MOTA naik** - Overall performance lebih baik

## Testing Checklist

Setelah perubahan ini, test:
- [ ] ID Switches < 100 (target: 64-75)
- [ ] FP < 1000 (target: 300-900)
- [ ] MOTA > 0.63 (target: 0.647+)
- [ ] IDF1 tetap tinggi (> 0.7)

Jika masih ada masalah:
- Bisa naikkan threshold lebih lagi (0.65 untuk Stage 2)
- Atau naikkan min_iou_thresh ke 0.35-0.4
- Atau turunkan ghost_match_thresh jika terlalu ketat

