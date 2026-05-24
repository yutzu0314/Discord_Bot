#!/usr/bin/env python
"""
Modul konfigurasi TrackGuard MOT.
Sekarang menggunakan satu sumber default di kode,
dan optional file JSON hanya jika tersedia.
"""

import json
import os
from typing import Dict, Any, Optional


class TrackGuardConfig:
    """
    Loader konfigurasi dengan fallback internal.
    """

    DEFAULT_CONFIG_PATH = "config/optimal_config.json"

    def __init__(self, config_path: Optional[str] = None):
        # Nilai default tuning khusus (dapat disesuaikan di sini)
        self.default_config: Dict[str, Any] = {
            # ===== TrackManager base parameters =====
            "max_age": 10,
            "min_hits": 4,
            "fps": 15,
            "max_track_history": 20,
            "min_confidence": 0.58,
            "max_track_id": 10000,

            # ===== Matching parameters =====
            "base_iou_threshold": 0.58,
            "new_track_iou_factor": 1.15,
            "static_iou_factor": 0.85,
            "high_det_thresh": 0.83,
            "low_det_thresh": 0.065,
            "match_thresh": 0.78,

            # ===== Static object detection =====
            "static_movement_threshold": 3.2502,
            "static_check_frames": 5,

            # ===== Confidence / filtering thresholds =====
            "edge_conf": 0.70,
            "small_edge_conf": 0.68,
            "small_conf": 0.65,
            "tiny_area_threshold": 0.0013,
            "edge_tiny_conf_threshold": 0.65,
            "small_obj_threshold": 0.0016,
            "edge_margin": 26,
            "small_edge_conf_reduction": 0.98,

            # ===== Confidence decay / retention =====
            "new_track_decay": 0.55,
            "static_decay_factor": 0.97,
            "edge_decay_penalty": 0.05,
            "max_decay_limit": 0.25,
            "edge_retention_factor": 0.70,
            "dynamic_base_decay": 0.76,
            "dynamic_conf_bonus": 0.14,
            "max_conf_factor": 1.8,
            "max_hits_factor": 3.0,
            "max_age_cap_factor": 1.5,
            "max_decay_age": 3,

            # ===== Confidence update (alpha) =====
            "static_edge_alpha": 0.78,
            "static_alpha": 0.60,
            "dynamic_base_alpha": 0.55,
            "dynamic_alpha_bonus": 0.18,
            "edge_alpha_factor": 0.74,

            # ===== Matching weights =====
            "static_iou_weight": 0.74,
            "static_color_weight": 0.16,
            "static_shape_weight": 0.10,
            "moving_iou_weight": 0.72,
            "moving_color_weight": 0.18,
            "moving_shape_weight": 0.10,
            "new_track_iou_weight": 0.97,
            "new_track_color_weight": 0.02,
            "new_track_shape_weight": 0.01,
        }

        self.config: Dict[str, Any] = self.default_config.copy()
        self.config_path: Optional[str] = None

        # Jika file konfigurasi diberikan dan ada, override default
        config_file = config_path or self.DEFAULT_CONFIG_PATH
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            if config_path:
                print(f"⚠️ Config file '{config_path}' tidak ditemukan. "
                      f"Menggunakan default internal.")

    def get(self, key: str, default: Any = None) -> Any:
        """Ambil nilai parameter dari konfigurasi."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set nilai parameter."""
        self.config[key] = value

    def update(self, params: Dict[str, Any]) -> None:
        """Update beberapa parameter sekaligus."""
        self.config.update(params)

    def load_config(self, config_path: str) -> None:
        """Muat konfigurasi dari file JSON."""
        with open(config_path, "r") as f:
            loaded_config = json.load(f)
            if not isinstance(loaded_config, dict):
                raise ValueError("File konfigurasi harus berupa JSON object (dict)")
            self.config.update(loaded_config)
            self.config_path = config_path
            print(f"✅ Config loaded from {config_path}")

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Simpan konfigurasi ke file JSON.
        """
        target_path = config_path or self.config_path
        try:
            with open(target_path, "w") as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Config saved to {target_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving config to {target_path}: {e}")
            return False

    def print_config(self) -> None:
        """Cetak konfigurasi saat ini."""
        print("==== TrackGuard Configuration ====")
        for key, value in self.config.items():
            print(f"{key}: {value}")

    def export_config_snapshot(self) -> Dict[str, Any]:
        """Return snapshot konfigurasi."""
        return self.config.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Ringkasan konfigurasi."""
        return {
            "total_parameters": len(self.config),
            "config_path": self.config_path,
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return f"TrackGuardConfig({summary['total_parameters']} parameters from {summary['config_path']})"

