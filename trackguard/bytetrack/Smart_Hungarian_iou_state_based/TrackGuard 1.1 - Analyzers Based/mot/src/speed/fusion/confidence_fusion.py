import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class ConfidenceFusion:
    """
    Confidence-Weighted Speed Fusion
    
    Modul inti yang mengimplementasi inovasi utama CWSF:
    pembobotan adaptif berdasarkan skor kepercayaan TrackGuard.
    """
    
    def __init__(self):
        """
        Initialize confidence fusion module
        """
        # Bobot dasar untuk setiap metode
        self.base_weights = {
            'pixel': 0.4,
            'trajectory': 0.35,
            'optical_flow': 0.25
        }
        
        # Ambang batas kepercayaan untuk strategi pembobotan
        self.confidence_thresholds = {
            'very_high': 0.9,    # Kepercayaan sangat tinggi
            'high': 0.75,        # Kepercayaan tinggi
            'medium': 0.5,       # Kepercayaan sedang  
            'low': 0.3,          # Kepercayaan rendah
            'very_low': 0.1      # Kepercayaan sangat rendah
        }
        
        # Parameter untuk adaptasi dinamis
        self.adaptation_parameters = {
            'stability_impact': 0.3,     # Pengaruh stabilitas tracking
            'occlusion_impact': 0.4,     # Pengaruh oklusi
            'speed_consistency_impact': 0.2,  # Pengaruh konsistensi kecepatan
            'trajectory_quality_impact': 0.1  # Pengaruh kualitas lintasan
        }
        
    def calculate_adaptive_weights(self, 
                                 trackguard_confidence: float,
                                 stability_factor: float = 1.0,
                                 occlusion_factor: float = 0.0,
                                 trajectory_confidence: float = 1.0,
                                 speed_consistency: float = 1.0) -> Dict[str, float]:
        """
        Hitung bobot adaptif berdasarkan berbagai faktor kualitas tracking
        
        Args:
            trackguard_confidence: Skor kepercayaan dari TrackGuard (0-1)
            stability_factor: Faktor stabilitas tracking (0-1)
            occlusion_factor: Faktor oklusi (0-1, 0=tidak ada oklusi)
            trajectory_confidence: Kepercayaan kualitas lintasan (0-1)
            speed_consistency: Konsistensi kecepatan historis (0-1)
            
        Returns:
            Dict[str, float]: Bobot adaptif untuk setiap metode
        """
        # Mulai dengan bobot dasar
        weights = self.base_weights.copy()
        
        # Hitung faktor gabungan kualitas tracking
        quality_score = self._calculate_overall_quality(
            trackguard_confidence, stability_factor, occlusion_factor,
            trajectory_confidence, speed_consistency
        )
        
        # Tentukan strategi pembobotan berdasarkan kualitas
        if quality_score >= self.confidence_thresholds['very_high']:
            weights = self._get_very_high_confidence_weights()
        elif quality_score >= self.confidence_thresholds['high']:
            weights = self._get_high_confidence_weights()
        elif quality_score >= self.confidence_thresholds['medium']:
            weights = self._get_medium_confidence_weights()
        elif quality_score >= self.confidence_thresholds['low']:
            weights = self._get_low_confidence_weights()
        else:
            weights = self._get_very_low_confidence_weights()
        
        # Terapkan penyesuaian berdasarkan kondisi spesifik
        weights = self._apply_contextual_adjustments(
            weights, occlusion_factor, trajectory_confidence, speed_consistency
        )
        
        # Normalisasi bobot agar total = 1
        weights = self._normalize_weights(weights)
        
        return weights
    
    def fuse_speed_estimates(self, 
                           speed_estimates: Dict[str, float],
                           weights: Dict[str, float],
                           confidence_scores: Optional[Dict[str, float]] = None) -> float:
        """
        Gabungkan estimasi kecepatan menggunakan bobot yang telah dihitung
        
        Args:
            speed_estimates: Estimasi kecepatan dari berbagai metode
            weights: Bobot untuk setiap metode
            confidence_scores: Skor kepercayaan untuk setiap metode (opsional)
            
        Returns:
            float: Kecepatan hasil fusion
        """
        if not speed_estimates or not weights:
            return 0.0
        
        fused_speed = 0.0
        total_weight = 0.0
        
        # Fusion berbobot dengan validasi
        for method, speed in speed_estimates.items():
            if method in weights and weights[method] > 0:
                weight = weights[method]
                
                # Terapkan faktor kepercayaan metode jika tersedia
                if confidence_scores and method in confidence_scores:
                    method_confidence = confidence_scores[method]
                    weight *= method_confidence
                
                # Validasi kecepatan (harus non-negatif dan masuk akal)
                if self._is_valid_speed(speed):
                    fused_speed += speed * weight
                    total_weight += weight
        
        # Normalisasi hasil
        if total_weight > 0:
            fused_speed /= total_weight
        
        return max(0.0, fused_speed)  # Pastikan tidak negatif
    
    def _calculate_overall_quality(self, 
                                  trackguard_confidence: float,
                                  stability_factor: float,
                                  occlusion_factor: float,
                                  trajectory_confidence: float,
                                  speed_consistency: float) -> float:
        """
        Hitung skor kualitas keseluruhan dari berbagai faktor
        """
        # Bobot untuk setiap faktor
        weights = [0.4, 0.25, 0.15, 0.1, 0.1]  # Total = 1.0
        factors = [
            trackguard_confidence,
            stability_factor,
            1.0 - occlusion_factor,  # Oklusi rendah = kualitas tinggi
            trajectory_confidence,
            speed_consistency
        ]
        
        # Hitung rata-rata berbobot
        quality_score = sum(w * f for w, f in zip(weights, factors))
        
        return max(0.0, min(1.0, quality_score))
    
    def _get_very_high_confidence_weights(self) -> Dict[str, float]:
        """Bobot untuk kepercayaan sangat tinggi - prioritas metode cepat"""
        return {
            'pixel': 0.8,           # Dominasi metode cepat
            'trajectory': 0.15,     # Sedikit validasi trajectory
            'optical_flow': 0.05    # Minimal optical flow
        }
    
    def _get_high_confidence_weights(self) -> Dict[str, float]:
        """Bobot untuk kepercayaan tinggi - seimbang condong ke metode cepat"""
        return {
            'pixel': 0.65,          # Masih dominan metode cepat
            'trajectory': 0.25,     # Lebih banyak trajectory
            'optical_flow': 0.1     # Sedikit optical flow
        }
    
    def _get_medium_confidence_weights(self) -> Dict[str, float]:
        """Bobot untuk kepercayaan sedang - seimbang"""
        return {
            'pixel': 0.45,          # Seimbang
            'trajectory': 0.35,     # Trajectory cukup berperan
            'optical_flow': 0.2     # Optical flow mulai signifikan
        }
    
    def _get_low_confidence_weights(self) -> Dict[str, float]:
        """Bobot untuk kepercayaan rendah - prioritas metode canggih"""
        return {
            'pixel': 0.25,          # Kurangi ketergantungan pixel
            'trajectory': 0.45,     # Dominasi trajectory
            'optical_flow': 0.3     # Signifikan optical flow
        }
    
    def _get_very_low_confidence_weights(self) -> Dict[str, float]:
        """Bobot untuk kepercayaan sangat rendah - dominasi metode canggih"""
        return {
            'pixel': 0.15,          # Minimal pixel
            'trajectory': 0.4,      # Tinggi trajectory
            'optical_flow': 0.45    # Dominasi optical flow
        }
    
    def _apply_contextual_adjustments(self,
                                    weights: Dict[str, float],
                                    occlusion_factor: float,
                                    trajectory_confidence: float,
                                    speed_consistency: float) -> Dict[str, float]:
        """
        Terapkan penyesuaian kontekstual berdasarkan kondisi spesifik
        """
        adjusted_weights = weights.copy()
        
        # Penyesuaian untuk oklusi tinggi
        if occlusion_factor > 0.5:
            # Kurangi ketergantungan pada pixel displacement
            adjusted_weights['pixel'] *= (1.0 - occlusion_factor * 0.6)
            # Tingkatkan bobot optical flow untuk menangani oklusi
            adjusted_weights['optical_flow'] *= (1.0 + occlusion_factor * 0.4)
        
        # Penyesuaian untuk kualitas trajectory rendah
        if trajectory_confidence < 0.5:
            # Kurangi bobot trajectory jika kualitasnya buruk
            reduction_factor = (1.0 - trajectory_confidence) * 0.5
            adjusted_weights['trajectory'] *= (1.0 - reduction_factor)
            # Redistribusi ke metode lain
            redistributed_weight = adjusted_weights['trajectory'] * reduction_factor
            adjusted_weights['pixel'] += redistributed_weight * 0.6
            adjusted_weights['optical_flow'] += redistributed_weight * 0.4
        
        # Penyesuaian untuk konsistensi kecepatan rendah
        if speed_consistency < 0.4:
            # Jika kecepatan tidak konsisten, lebih andalkan optical flow
            consistency_penalty = (0.4 - speed_consistency) * 0.3
            adjusted_weights['pixel'] *= (1.0 - consistency_penalty)
            adjusted_weights['optical_flow'] *= (1.0 + consistency_penalty)
        
        return adjusted_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalisasi bobot agar totalnya = 1.0
        """
        total_weight = sum(weights.values())
        
        if total_weight <= 0:
            # Jika semua bobot 0, gunakan bobot seimbang
            num_methods = len(weights)
            return {method: 1.0 / num_methods for method in weights.keys()}
        
        # Normalisasi
        normalized_weights = {
            method: weight / total_weight 
            for method, weight in weights.items()
        }
        
        return normalized_weights
    
    def _is_valid_speed(self, speed: float) -> bool:
        """
        Validasi apakah nilai kecepatan masuk akal
        
        Args:
            speed: Nilai kecepatan untuk divalidasi
            
        Returns:
            bool: True jika kecepatan valid
        """
        # Kecepatan harus non-negatif dan tidak terlalu ekstrem
        if speed < 0:
            return False
        
        # Batas maksimum kecepatan yang masuk akal (misalnya 200 km/h = 55.6 m/s)
        max_reasonable_speed = 60.0  # m/s
        if speed > max_reasonable_speed:
            return False
        
        # Periksa apakah bukan NaN atau infinity
        if math.isnan(speed) or math.isinf(speed):
            return False
        
        return True
    
    def calculate_fusion_confidence(self,
                                  speed_estimates: Dict[str, float],
                                  weights: Dict[str, float],
                                  method_confidences: Optional[Dict[str, float]] = None) -> float:
        """
        Hitung tingkat kepercayaan hasil fusion
        
        Args:
            speed_estimates: Estimasi kecepatan dari berbagai metode
            weights: Bobot yang digunakan untuk fusion
            method_confidences: Kepercayaan masing-masing metode
            
        Returns:
            float: Skor kepercayaan fusion (0-1)
        """
        if not speed_estimates or len(speed_estimates) < 2:
            return 0.0
        
        # Faktor 1: Konsistensi antar metode
        consistency_score = self._calculate_method_consistency(speed_estimates)
        
        # Faktor 2: Distribusi bobot (bobot yang merata = kepercayaan tinggi)
        weight_distribution_score = self._calculate_weight_distribution_score(weights)
        
        # Faktor 3: Kepercayaan rata-rata metode
        if method_confidences:
            avg_method_confidence = np.mean([
                conf for method, conf in method_confidences.items()
                if method in speed_estimates
            ])
        else:
            avg_method_confidence = 0.5  # Default jika tidak ada info
        
        # Gabungkan semua faktor
        fusion_confidence = (
            consistency_score * 0.4 +
            weight_distribution_score * 0.3 +
            avg_method_confidence * 0.3
        )
        
        return max(0.0, min(1.0, fusion_confidence))
    
    def _calculate_method_consistency(self, speed_estimates: Dict[str, float]) -> float:
        """
        Hitung konsistensi antar metode estimasi
        """
        speeds = list(speed_estimates.values())
        
        if len(speeds) < 2:
            return 1.0
        
        # Hitung koefisien variasi
        mean_speed = np.mean(speeds)
        if mean_speed == 0:
            return 1.0 if all(s == 0 for s in speeds) else 0.0
        
        std_speed = np.std(speeds)
        cv = std_speed / mean_speed
        
        # Konversi ke skor konsistensi (0-1)
        # CV rendah = konsistensi tinggi
        consistency = math.exp(-cv * 2)  # Eksponensial negatif
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_weight_distribution_score(self, weights: Dict[str, float]) -> float:
        """
        Hitung skor distribusi bobot (seimbang = lebih baik)
        """
        weight_values = list(weights.values())
        
        if not weight_values:
            return 0.0
        
        # Hitung entropi Shannon untuk mengukur distribusi
        # Distribusi seimbang memiliki entropi tinggi
        entropy = 0.0
        for weight in weight_values:
            if weight > 0:
                entropy -= weight * math.log2(weight)
        
        # Normalisasi entropi (max entropy untuk distribusi seimbang)
        max_entropy = math.log2(len(weight_values))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return max(0.0, min(1.0, normalized_entropy))
    
    def get_fusion_summary(self,
                          speed_estimates: Dict[str, float],
                          weights: Dict[str, float],
                          fused_speed: float) -> Dict:
        """
        Dapatkan ringkasan lengkap proses fusion
        
        Args:
            speed_estimates: Estimasi kecepatan dari berbagai metode
            weights: Bobot yang digunakan
            fused_speed: Hasil kecepatan fusion
            
        Returns:
            Dict: Ringkasan fusion
        """
        return {
            'fused_speed': fused_speed,
            'method_estimates': speed_estimates.copy(),
            'weights_used': weights.copy(),
            'fusion_confidence': self.calculate_fusion_confidence(speed_estimates, weights),
            'method_consistency': self._calculate_method_consistency(speed_estimates),
            'weight_distribution_score': self._calculate_weight_distribution_score(weights),
            'dominant_method': max(weights, key=weights.get) if weights else None,
            'num_methods_used': len([w for w in weights.values() if w > 0.01])
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Perbarui ambang batas kepercayaan
        
        Args:
            new_thresholds: Ambang batas baru
        """
        self.confidence_thresholds.update(new_thresholds)
    
    def update_base_weights(self, new_weights: Dict[str, float]):
        """
        Perbarui bobot dasar
        
        Args:
            new_weights: Bobot dasar baru
        """
        # Validasi dan normalisasi bobot baru
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            normalized_weights = {
                method: weight / total_weight
                for method, weight in new_weights.items()
            }
            self.base_weights.update(normalized_weights)