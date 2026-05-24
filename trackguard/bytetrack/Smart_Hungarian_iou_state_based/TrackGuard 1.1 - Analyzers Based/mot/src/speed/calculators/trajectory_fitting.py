import numpy as np
from typing import List, Tuple, Optional, Dict
import math
from scipy.optimize import curve_fit
from collections import deque

class TrajectoryFittingCalculator:
    """
    Trajectory-based speed calculation method
    
    Menganalisis lintasan pergerakan objek untuk menghitung kecepatan
    berdasarkan model matematis lintasan. Metode ini lebih akurat
    untuk objek yang bergerak dalam pola yang dapat diprediksi.
    """
    
    def __init__(self, 
                 pixels_per_meter: float = 20.0,
                 fps: float = 30.0,
                 min_history: int = 3,
                 max_history: int = 10):
        """
        Initialize trajectory fitting calculator
        
        Args:
            pixels_per_meter: Parameter kalibrasi kamera (piksel per meter)
            fps: Frame per detik video
            min_history: Jumlah minimum titik untuk fitting
            max_history: Jumlah maksimum titik yang disimpan
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.min_history = min_history
        self.max_history = max_history
        
        # Menyimpan riwayat posisi untuk setiap track
        self.position_history = {}
        self.time_history = {}
        
    def calculate_speed(self, track_id: int, bbox_current: List[float], 
                       frame_number: int) -> float:
        """
        Hitung kecepatan berdasarkan trajectory fitting
        
        Args:
            track_id: ID track untuk menyimpan riwayat
            bbox_current: Bounding box saat ini [x1, y1, x2, y2]
            frame_number: Nomor frame saat ini
            
        Returns:
            float: Kecepatan dalam m/s
        """
        # Ambil titik tengah bbox
        center = self._get_bbox_center(bbox_current)
        current_time = frame_number * self.frame_time
        
        # Simpan ke riwayat
        self._update_history(track_id, center, current_time)
        
        # Cek apakah cukup data untuk fitting
        if (track_id not in self.position_history or 
            len(self.position_history[track_id]) < self.min_history):
            return 0.0
        
        # Lakukan trajectory fitting dan hitung kecepatan
        try:
            speed = self._fit_trajectory_and_calculate_speed(track_id)
            return max(0.0, speed)  # Pastikan tidak negatif
        except Exception:
            # Jika fitting gagal, fallback ke metode sederhana
            return self._fallback_speed_calculation(track_id)
    
    def calculate_speed_kmh(self, track_id: int, bbox_current: List[float], 
                           frame_number: int) -> float:
        """
        Hitung kecepatan dalam km/h
        
        Args:
            track_id: ID track
            bbox_current: Bounding box saat ini
            frame_number: Nomor frame
            
        Returns:
            float: Kecepatan dalam km/h
        """
        speed_ms = self.calculate_speed(track_id, bbox_current, frame_number)
        return speed_ms * 3.6
    
    def get_trajectory_confidence(self, track_id: int) -> float:
        """
        Hitung tingkat kepercayaan trajectory fitting
        
        Args:
            track_id: ID track
            
        Returns:
            float: Skor kepercayaan (0-1)
        """
        if track_id not in self.position_history:
            return 0.0
        
        positions = list(self.position_history[track_id])
        if len(positions) < self.min_history:
            return 0.0
        
        try:
            # Hitung R-squared untuk menilai kualitas fitting
            r_squared = self._calculate_fitting_quality(track_id)
            
            # Hitung konsistensi arah pergerakan
            direction_consistency = self._calculate_direction_consistency(track_id)
            
            # Hitung kestabilan kecepatan
            speed_stability = self._calculate_speed_stability(track_id)
            
            # Gabungkan semua faktor
            confidence = (r_squared * 0.5 + 
                         direction_consistency * 0.3 + 
                         speed_stability * 0.2)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    def _update_history(self, track_id: int, position: Tuple[float, float], 
                       time: float):
        """Perbarui riwayat posisi dan waktu"""
        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=self.max_history)
            self.time_history[track_id] = deque(maxlen=self.max_history)
        
        self.position_history[track_id].append(position)
        self.time_history[track_id].append(time)
    
    def _fit_trajectory_and_calculate_speed(self, track_id: int) -> float:
        """
        Lakukan trajectory fitting dan hitung kecepatan
        """
        positions = list(self.position_history[track_id])
        times = list(self.time_history[track_id])
        
        if len(positions) < self.min_history:
            return 0.0
        
        # Konversi posisi ke array numpy
        x_positions = np.array([pos[0] for pos in positions])
        y_positions = np.array([pos[1] for pos in positions])
        time_array = np.array(times)
        
        # Normalisasi waktu mulai dari 0
        time_normalized = time_array - time_array[0]
        
        # Tentukan model fitting berdasarkan jumlah data
        if len(positions) <= 4:
            # Model linear untuk data terbatas
            speed_x = self._fit_linear_model(time_normalized, x_positions)
            speed_y = self._fit_linear_model(time_normalized, y_positions)
        else:
            # Model polinomial untuk data lebih banyak
            speed_x = self._fit_polynomial_model(time_normalized, x_positions)
            speed_y = self._fit_polynomial_model(time_normalized, y_positions)
        
        # Hitung kecepatan total dalam piksel/detik
        speed_pixels_per_second = math.sqrt(speed_x**2 + speed_y**2)
        
        # Konversi ke m/s
        speed_ms = speed_pixels_per_second / self.pixels_per_meter
        
        return speed_ms
    
    def _fit_linear_model(self, time_array: np.ndarray, 
                         position_array: np.ndarray) -> float:
        """
        Fitting model linear: position = a*t + b
        Return kecepatan (turunan = a)
        """
        try:
            # Gunakan least squares untuk fitting linear
            coeffs = np.polyfit(time_array, position_array, 1)
            speed = coeffs[0]  # Koefisien t (turunan)
            return speed
        except Exception:
            # Fallback ke perbedaan sederhana
            if len(position_array) >= 2:
                return (position_array[-1] - position_array[0]) / (time_array[-1] - time_array[0])
            return 0.0
    
    def _fit_polynomial_model(self, time_array: np.ndarray, 
                             position_array: np.ndarray) -> float:
        """
        Fitting model polinomial orde 2: position = a*t^2 + b*t + c
        Return kecepatan pada titik terakhir (turunan = 2*a*t + b)
        """
        try:
            # Fitting polinomial orde 2
            coeffs = np.polyfit(time_array, position_array, 2)
            a, b, c = coeffs
            
            # Hitung turunan pada titik terakhir
            last_time = time_array[-1]
            speed = 2 * a * last_time + b
            
            return speed
        except Exception:
            # Fallback ke model linear
            return self._fit_linear_model(time_array, position_array)
    
    def _calculate_fitting_quality(self, track_id: int) -> float:
        """
        Hitung kualitas fitting menggunakan R-squared
        """
        try:
            positions = list(self.position_history[track_id])
            times = list(self.time_history[track_id])
            
            x_positions = np.array([pos[0] for pos in positions])
            time_array = np.array(times)
            time_normalized = time_array - time_array[0]
            
            # Fitting linear
            coeffs = np.polyfit(time_normalized, x_positions, 1)
            predicted = np.polyval(coeffs, time_normalized)
            
            # Hitung R-squared
            ss_res = np.sum((x_positions - predicted) ** 2)
            ss_tot = np.sum((x_positions - np.mean(x_positions)) ** 2)
            
            if ss_tot == 0:
                return 1.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, r_squared)
            
        except Exception:
            return 0.0
    
    def _calculate_direction_consistency(self, track_id: int) -> float:
        """
        Hitung konsistensi arah pergerakan
        """
        try:
            positions = list(self.position_history[track_id])
            if len(positions) < 3:
                return 1.0
            
            # Hitung sudut antara segmen berturut-turut
            angles = []
            for i in range(len(positions) - 2):
                p1, p2, p3 = positions[i], positions[i+1], positions[i+2]
                
                # Vektor segmen
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Hitung sudut antara vektor
                angle = self._angle_between_vectors(v1, v2)
                angles.append(angle)
            
            if not angles:
                return 1.0
            
            # Konsistensi tinggi jika sudut kecil (arah tidak berubah drastis)
            avg_angle = np.mean(angles)
            consistency = 1.0 - (avg_angle / 180.0)  # Normalisasi ke 0-1
            
            return max(0.0, consistency)
            
        except Exception:
            return 0.0
    
    def _calculate_speed_stability(self, track_id: int) -> float:
        """
        Hitung kestabilan kecepatan antar segmen
        """
        try:
            positions = list(self.position_history[track_id])
            times = list(self.time_history[track_id])
            
            if len(positions) < 3:
                return 1.0
            
            # Hitung kecepatan antar segmen
            speeds = []
            for i in range(len(positions) - 1):
                p1, p2 = positions[i], positions[i+1]
                t1, t2 = times[i], times[i+1]
                
                distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                time_diff = t2 - t1
                
                if time_diff > 0:
                    speed = distance / time_diff
                    speeds.append(speed)
            
            if len(speeds) < 2:
                return 1.0
            
            # Hitung koefisien variasi (stabilitas)
            mean_speed = np.mean(speeds)
            if mean_speed == 0:
                return 1.0
            
            std_speed = np.std(speeds)
            cv = std_speed / mean_speed
            
            # Konversi ke skor stabilitas (0-1)
            stability = math.exp(-cv)  # Eksponensial negatif
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.0
    
    def _angle_between_vectors(self, v1: Tuple[float, float], 
                              v2: Tuple[float, float]) -> float:
        """
        Hitung sudut antara dua vektor dalam derajat
        """
        try:
            # Normalisasi vektor
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 == 0 or mag2 == 0:
                return 0.0
            
            # Hitung cosinus sudut
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = dot_product / (mag1 * mag2)
            
            # Batasi nilai untuk menghindari domain error
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            # Konversi ke derajat
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
            
        except Exception:
            return 0.0
    
    def _fallback_speed_calculation(self, track_id: int) -> float:
        """
        Perhitungan kecepatan cadangan jika fitting gagal
        """
        try:
            positions = list(self.position_history[track_id])
            times = list(self.time_history[track_id])
            
            if len(positions) < 2:
                return 0.0
            
            # Gunakan dua titik terakhir
            p1, p2 = positions[-2], positions[-1]
            t1, t2 = times[-2], times[-1]
            
            # Hitung jarak dan waktu
            distance_pixels = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            time_diff = t2 - t1
            
            if time_diff <= 0:
                return 0.0
            
            # Konversi ke m/s
            speed_pixels_per_second = distance_pixels / time_diff
            speed_ms = speed_pixels_per_second / self.pixels_per_meter
            
            return speed_ms
            
        except Exception:
            return 0.0
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Hitung titik tengah bounding box
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return center_x, center_y
    
    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> int:
        """
        Bersihkan data untuk track yang tidak aktif
        
        Args:
            active_track_ids: Daftar ID track yang masih aktif
            
        Returns:
            int: Jumlah track yang dibersihkan
        """
        all_tracks = set(self.position_history.keys())
        active_tracks = set(active_track_ids)
        stale_tracks = all_tracks - active_tracks
        
        # Bersihkan track yang tidak aktif
        for track_id in stale_tracks:
            if track_id in self.position_history:
                del self.position_history[track_id]
            if track_id in self.time_history:
                del self.time_history[track_id]
        
        return len(stale_tracks)
    
    def update_calibration(self, pixels_per_meter: float, fps: float = None):
        """
        Perbarui parameter kalibrasi
        
        Args:
            pixels_per_meter: Parameter kalibrasi baru
            fps: Nilai fps baru (opsional)
        """
        self.pixels_per_meter = pixels_per_meter
        if fps is not None:
            self.fps = fps
            self.frame_time = 1.0 / fps
    
    def get_trajectory_summary(self, track_id: int) -> Optional[Dict]:
        """
        Dapatkan ringkasan lintasan untuk suatu track
        
        Args:
            track_id: ID track
            
        Returns:
            Optional[Dict]: Ringkasan lintasan atau None jika tidak ditemukan
        """
        if track_id not in self.position_history:
            return None
        
        positions = list(self.position_history[track_id])
        times = list(self.time_history[track_id])
        
        if not positions:
            return None
        
        # Hitung total jarak tempuh
        total_distance_pixels = 0.0
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i+1]
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance_pixels += distance
        
        total_distance_meters = total_distance_pixels / self.pixels_per_meter
        
        return {
            'track_id': track_id,
            'total_points': len(positions),
            'total_distance_pixels': total_distance_pixels,
            'total_distance_meters': total_distance_meters,
            'total_time': times[-1] - times[0] if len(times) > 1 else 0.0,
            'trajectory_confidence': self.get_trajectory_confidence(track_id),
            'start_position': positions[0],
            'end_position': positions[-1]
        }