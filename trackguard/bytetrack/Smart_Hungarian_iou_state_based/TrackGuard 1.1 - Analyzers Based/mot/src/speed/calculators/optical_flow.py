import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import math
from collections import deque

class OpticalFlowCalculator:
    """
    Optical flow-based speed calculation method
    
    Menganalisis pergerakan piksel individual untuk mendeteksi kecepatan
    objek yang mengalami oklusi parsial atau perubahan bentuk kompleks.
    Method ini paling akurat tapi juga paling computationally expensive.
    """
    
    def __init__(self, 
                 pixels_per_meter: float = 20.0,
                 fps: float = 30.0,
                 flow_method: str = 'lucas_kanade',
                 feature_params: Optional[Dict] = None,
                 lk_params: Optional[Dict] = None):
        """
        Initialize optical flow calculator
        
        Args:
            pixels_per_meter: Parameter kalibrasi kamera (piksel per meter)
            fps: Frame per detik video
            flow_method: Metode optical flow ('lucas_kanade' atau 'farneback')
            feature_params: Parameter untuk deteksi corner/feature points
            lk_params: Parameter untuk Lucas-Kanade optical flow
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.flow_method = flow_method
        
        # Default parameters untuk Lucas-Kanade
        self.feature_params = feature_params or {
            'maxCorners': 100,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }
        
        self.lk_params = lk_params or {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # Default parameters untuk Farneback
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Storage untuk frame dan feature tracking
        self.previous_frames = {}  # track_id -> previous gray frame
        self.previous_features = {}  # track_id -> previous feature points
        self.flow_history = {}  # track_id -> deque of flow vectors
        self.roi_history = {}  # track_id -> deque of ROI regions
        self.confidence_scores = {}  # track_id -> confidence scores
        
        # Configuration
        self.max_flow_history = 5
        self.min_features_threshold = 10  # Minimum features untuk reliable tracking
        self.max_flow_magnitude = 50  # Maximum reasonable flow magnitude
        
    def calculate_speed(self, track_id: int, bbox_current: List[float], 
                       frame_current: np.ndarray, frame_number: int) -> float:
        """
        Hitung kecepatan menggunakan optical flow
        
        Args:
            track_id: ID track untuk menyimpan riwayat
            bbox_current: Bounding box saat ini [x1, y1, x2, y2]
            frame_current: Frame video saat ini (BGR)
            frame_number: Nomor frame saat ini
            
        Returns:
            float: Kecepatan dalam m/s
        """
        # Konversi ke grayscale
        gray_current = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY)
        
        # Extract ROI dari frame current
        roi_current = self._extract_roi(gray_current, bbox_current)
        
        # Update ROI history
        self._update_roi_history(track_id, bbox_current)
        
        # Cek apakah ada frame sebelumnya
        if track_id not in self.previous_frames:
            # First frame untuk track ini
            self._initialize_track(track_id, gray_current, roi_current, bbox_current)
            return 0.0
        
        # Ambil frame dan features sebelumnya
        gray_previous = self.previous_frames[track_id]
        roi_previous = self._extract_roi(gray_previous, self.roi_history[track_id][-2])
        
        try:
            # Hitung optical flow berdasarkan metode yang dipilih
            if self.flow_method == 'lucas_kanade':
                flow_vector = self._calculate_lucas_kanade_flow(
                    track_id, roi_previous, roi_current, bbox_current
                )
            elif self.flow_method == 'farneback':
                flow_vector = self._calculate_farneback_flow(
                    roi_previous, roi_current
                )
            else:
                raise ValueError(f"Unknown flow method: {self.flow_method}")
            
            # Konversi flow vector ke kecepatan
            speed = self._flow_to_speed(flow_vector)
            
            # Update histories
            self._update_flow_history(track_id, flow_vector)
            self._update_confidence_score(track_id, flow_vector, roi_current)
            
            # Update stored frame dan features untuk frame berikutnya
            self._update_track_data(track_id, gray_current, roi_current, bbox_current)
            
            return max(0.0, speed)  # Pastikan tidak negatif
            
        except Exception as e:
            # Jika optical flow gagal, fallback ke metode sederhana
            return self._fallback_speed_calculation(track_id, bbox_current)
    
    def calculate_speed_kmh(self, track_id: int, bbox_current: List[float], 
                           frame_current: np.ndarray, frame_number: int) -> float:
        """
        Hitung kecepatan dalam km/h
        
        Args:
            track_id: ID track
            bbox_current: Bounding box saat ini
            frame_current: Frame video saat ini
            frame_number: Nomor frame
            
        Returns:
            float: Kecepatan dalam km/h
        """
        speed_ms = self.calculate_speed(track_id, bbox_current, frame_current, frame_number)
        return speed_ms * 3.6
    
    def get_flow_confidence(self, track_id: int) -> float:
        """
        Hitung tingkat kepercayaan optical flow
        
        Args:
            track_id: ID track
            
        Returns:
            float: Skor kepercayaan (0-1)
        """
        if track_id not in self.confidence_scores:
            return 0.0
        
        return self.confidence_scores[track_id]
    
    def _initialize_track(self, track_id: int, gray_frame: np.ndarray, 
                         roi: np.ndarray, bbox: List[float]):
        """Initialize tracking data untuk track baru"""
        self.previous_frames[track_id] = gray_frame.copy()
        self.flow_history[track_id] = deque(maxlen=self.max_flow_history)
        self.roi_history[track_id] = deque(maxlen=self.max_flow_history)
        self.confidence_scores[track_id] = 0.0
        
        # Initialize features untuk Lucas-Kanade
        if self.flow_method == 'lucas_kanade':
            features = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
            self.previous_features[track_id] = features
    
    def _extract_roi(self, gray_frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract region of interest dari frame"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox dalam batas frame
        height, width = gray_frame.shape
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(x1+1, min(x2, width))
        y2 = max(y1+1, min(y2, height))
        
        return gray_frame[y1:y2, x1:x2]
    
    def _calculate_lucas_kanade_flow(self, track_id: int, roi_previous: np.ndarray, 
                                   roi_current: np.ndarray, bbox_current: List[float]) -> Tuple[float, float]:
        """
        Hitung optical flow menggunakan Lucas-Kanade method
        """
        # Ambil features sebelumnya
        if track_id not in self.previous_features or self.previous_features[track_id] is None:
            # Detect features di ROI sebelumnya
            features_prev = cv2.goodFeaturesToTrack(roi_previous, mask=None, **self.feature_params)
        else:
            features_prev = self.previous_features[track_id]
        
        if features_prev is None or len(features_prev) < self.min_features_threshold:
            # Tidak cukup features untuk tracking
            return (0.0, 0.0)
        
        # Hitung optical flow
        features_next, status, error = cv2.calcOpticalFlowPyrLK(
            roi_previous, roi_current, features_prev, None, **self.lk_params
        )
        
        # Filter good features (status = 1)
        good_new = features_next[status == 1]
        good_old = features_prev[status == 1]
        
        if len(good_new) < self.min_features_threshold:
            # Tidak cukup good features
            return (0.0, 0.0)
        
        # Hitung rata-rata flow vector
        flow_vectors = good_new - good_old
        
        # Filter outliers (flow yang terlalu besar)
        valid_flows = []
        for flow in flow_vectors:
            magnitude = np.linalg.norm(flow)
            if magnitude < self.max_flow_magnitude:
                valid_flows.append(flow)
        
        if not valid_flows:
            return (0.0, 0.0)
        
        # Hitung median flow untuk mengurangi noise
        valid_flows = np.array(valid_flows)
        median_flow = np.median(valid_flows, axis=0)
        
        # Update features untuk frame berikutnya
        new_features = cv2.goodFeaturesToTrack(roi_current, mask=None, **self.feature_params)
        self.previous_features[track_id] = new_features
        
        return tuple(median_flow)
    
    def _calculate_farneback_flow(self, roi_previous: np.ndarray, 
                                roi_current: np.ndarray) -> Tuple[float, float]:
        """
        Hitung optical flow menggunakan Farneback method
        """
        # Hitung dense optical flow
        flow = cv2.calcOpticalFlowPyrLK(roi_previous, roi_current, None, **self.farneback_params)
        
        # Hitung magnitude dan angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Filter pixels dengan flow magnitude yang reasonable
        valid_mask = (magnitude > 0.5) & (magnitude < self.max_flow_magnitude)
        
        if not np.any(valid_mask):
            return (0.0, 0.0)
        
        # Hitung rata-rata flow vector dari valid pixels
        valid_flow_x = flow[..., 0][valid_mask]
        valid_flow_y = flow[..., 1][valid_mask]
        
        mean_flow_x = np.median(valid_flow_x)
        mean_flow_y = np.median(valid_flow_y)
        
        return (mean_flow_x, mean_flow_y)
    
    def _flow_to_speed(self, flow_vector: Tuple[float, float]) -> float:
        """
        Konversi flow vector ke kecepatan dalam m/s
        """
        flow_x, flow_y = flow_vector
        
        # Hitung magnitude flow dalam piksel per frame
        flow_magnitude = math.sqrt(flow_x**2 + flow_y**2)
        
        # Konversi ke piksel per detik
        pixels_per_second = flow_magnitude * self.fps
        
        # Konversi ke m/s
        speed_ms = pixels_per_second / self.pixels_per_meter
        
        return speed_ms
    
    def _update_flow_history(self, track_id: int, flow_vector: Tuple[float, float]):
        """Update history flow vectors"""
        if track_id not in self.flow_history:
            self.flow_history[track_id] = deque(maxlen=self.max_flow_history)
        
        self.flow_history[track_id].append(flow_vector)
    
    def _update_roi_history(self, track_id: int, bbox: List[float]):
        """Update history ROI bounding boxes"""
        if track_id not in self.roi_history:
            self.roi_history[track_id] = deque(maxlen=self.max_flow_history)
        
        self.roi_history[track_id].append(bbox)
    
    def _update_confidence_score(self, track_id: int, flow_vector: Tuple[float, float], 
                               roi_current: np.ndarray):
        """
        Update confidence score berdasarkan kualitas optical flow
        """
        flow_x, flow_y = flow_vector
        flow_magnitude = math.sqrt(flow_x**2 + flow_y**2)
        
        # Faktor 1: Flow magnitude (reasonable range)
        magnitude_score = 1.0 if flow_magnitude < self.max_flow_magnitude else 0.0
        
        # Faktor 2: Konsistensi dengan flow history
        consistency_score = self._calculate_flow_consistency(track_id, flow_vector)
        
        # Faktor 3: Jumlah features yang berhasil di-track (untuk Lucas-Kanade)
        feature_score = 1.0  # Default untuk Farneback
        if (self.flow_method == 'lucas_kanade' and 
            track_id in self.previous_features and 
            self.previous_features[track_id] is not None):
            feature_count = len(self.previous_features[track_id])
            feature_score = min(1.0, feature_count / self.min_features_threshold)
        
        # Faktor 4: ROI quality (tidak terlalu kecil)
        roi_area = roi_current.shape[0] * roi_current.shape[1]
        min_roi_area = 100  # piksel
        roi_score = min(1.0, roi_area / min_roi_area)
        
        # Gabungkan semua faktor
        confidence = (magnitude_score * 0.3 + 
                     consistency_score * 0.4 + 
                     feature_score * 0.2 + 
                     roi_score * 0.1)
        
        self.confidence_scores[track_id] = max(0.0, min(1.0, confidence))
    
    def _calculate_flow_consistency(self, track_id: int, 
                                  current_flow: Tuple[float, float]) -> float:
        """
        Hitung konsistensi flow dengan history
        """
        if (track_id not in self.flow_history or 
            len(self.flow_history[track_id]) < 2):
            return 1.0  # Default untuk flow pertama
        
        flow_history = list(self.flow_history[track_id])
        current_magnitude = math.sqrt(current_flow[0]**2 + current_flow[1]**2)
        
        # Hitung rata-rata magnitude dari history
        magnitudes = [math.sqrt(fx**2 + fy**2) for fx, fy in flow_history]
        mean_magnitude = np.mean(magnitudes)
        
        if mean_magnitude == 0:
            return 1.0 if current_magnitude == 0 else 0.0
        
        # Hitung consistency sebagai inverse dari coefficient of variation
        std_magnitude = np.std(magnitudes + [current_magnitude])
        cv = std_magnitude / (mean_magnitude + 1e-6)
        
        consistency = math.exp(-cv)  # Eksponensial decay
        return max(0.0, min(1.0, consistency))
    
    def _update_track_data(self, track_id: int, gray_frame: np.ndarray, 
                         roi_current: np.ndarray, bbox_current: List[float]):
        """Update data untuk frame berikutnya"""
        self.previous_frames[track_id] = gray_frame.copy()
    
    def _fallback_speed_calculation(self, track_id: int, 
                                  bbox_current: List[float]) -> float:
        """
        Perhitungan kecepatan cadangan jika optical flow gagal
        """
        if (track_id not in self.roi_history or 
            len(self.roi_history[track_id]) < 2):
            return 0.0
        
        # Gunakan perpindahan center point sebagai fallback
        bbox_previous = self.roi_history[track_id][-2]
        
        # Hitung center points
        center_current = ((bbox_current[0] + bbox_current[2]) / 2,
                         (bbox_current[1] + bbox_current[3]) / 2)
        center_previous = ((bbox_previous[0] + bbox_previous[2]) / 2,
                          (bbox_previous[1] + bbox_previous[3]) / 2)
        
        # Hitung perpindahan
        dx = center_current[0] - center_previous[0]
        dy = center_current[1] - center_previous[1]
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # Konversi ke m/s
        pixels_per_second = distance_pixels * self.fps
        speed_ms = pixels_per_second / self.pixels_per_meter
        
        return speed_ms
    
    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> int:
        """
        Bersihkan data untuk track yang tidak aktif
        
        Args:
            active_track_ids: Daftar ID track yang masih aktif
            
        Returns:
            int: Jumlah track yang dibersihkan
        """
        all_tracks = set(self.previous_frames.keys())
        active_tracks = set(active_track_ids)
        stale_tracks = all_tracks - active_tracks
        
        # Bersihkan semua data untuk stale tracks
        for track_id in stale_tracks:
            if track_id in self.previous_frames:
                del self.previous_frames[track_id]
            if track_id in self.previous_features:
                del self.previous_features[track_id]
            if track_id in self.flow_history:
                del self.flow_history[track_id]
            if track_id in self.roi_history:
                del self.roi_history[track_id]
            if track_id in self.confidence_scores:
                del self.confidence_scores[track_id]
        
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
    
    def get_flow_summary(self, track_id: int) -> Optional[Dict]:
        """
        Dapatkan ringkasan optical flow untuk suatu track
        
        Args:
            track_id: ID track
            
        Returns:
            Optional[Dict]: Ringkasan optical flow atau None jika tidak ditemukan
        """
        if track_id not in self.flow_history:
            return None
        
        flow_vectors = list(self.flow_history[track_id])
        if not flow_vectors:
            return None
        
        # Hitung statistik flow
        magnitudes = [math.sqrt(fx**2 + fy**2) for fx, fy in flow_vectors]
        
        return {
            'track_id': track_id,
            'flow_method': self.flow_method,
            'total_flows': len(flow_vectors),
            'avg_flow_magnitude': np.mean(magnitudes) if magnitudes else 0,
            'max_flow_magnitude': max(magnitudes) if magnitudes else 0,
            'flow_consistency': self._calculate_flow_consistency(track_id, flow_vectors[-1]) if flow_vectors else 0,
            'confidence_score': self.confidence_scores.get(track_id, 0),
            'recent_flow_vector': flow_vectors[-1] if flow_vectors else (0, 0)
        }
    
    def set_flow_method(self, method: str):
        """
        Ubah metode optical flow
        
        Args:
            method: 'lucas_kanade' atau 'farneback'
        """
        if method in ['lucas_kanade', 'farneback']:
            self.flow_method = method
            # Reset semua tracking data karena metode berubah
            self.previous_frames.clear()
            self.previous_features.clear()
            self.flow_history.clear()
            self.roi_history.clear()
            self.confidence_scores.clear()
        else:
            raise ValueError(f"Unknown flow method: {method}")