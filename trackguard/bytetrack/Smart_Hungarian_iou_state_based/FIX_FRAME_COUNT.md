# Fix: frame_count AttributeError

## Problem
```
AttributeError: 'PureSmartHungarianTrackManager' object has no attribute 'frame_count'
```

## Root Cause
Saat cleanup/refactoring, `frame_count` dan `pipeline_stats` tidak di-initialize di `__init__` method.

## Solution
Ditambahkan initialization di `__init__`:

```python
# Performance tracking
self.frame_count = 0
self.pipeline_stats = {
    'detection_time': [],
    'iou_calculation_time': [],
    'data_association_time': [],
    'track_update_time': [],
    'total_pipeline_time': []
}
```

## Status
✅ Fixed - `frame_count` dan `pipeline_stats` sudah di-initialize

