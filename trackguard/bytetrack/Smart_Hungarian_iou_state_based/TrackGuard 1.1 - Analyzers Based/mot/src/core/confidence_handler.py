from typing import List, Dict
import numpy as np
from math import exp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrackGuard.ConfidenceHandler")

class ConfidenceHandler:
    """Enhanced confidence handling with adaptive fusion, history management and bonus"""
    
    def __init__(self):
        # Initialize base weights - lebih seimbang untuk komponen
        self.base_weights = {
            'yolo': 0.45,  # Slightly reduced from 0.5
            'color': 0.3,  # Increased from 0.25
            'shape': 0.25  # Unchanged
        }
        
        # History management parameters
        self.αbase = 0.35  # Ditingkatkan dari 0.3
        self.αmax = 0.7
        self.history_confidence = {}  # track_id -> history confidence
        
        # History bonus parameters - ditingkatkan untuk track stabil
        self.hmax = 0.35  # Ditingkatkan dari 0.3
        
        # ACF adaptation parameters
        self.acf_max_weight = 0.75    # Ditingkatkan dari 0.7
        self.acf_min_weight = 0.15    # Ditingkatkan dari 0.1
        self.stability_impact = 0.45  # Diturunkan dari 0.5
        self.maturity_impact = 0.25   # Diturunkan dari 0.3
        
        # Logging
        self.debug_level = 0
        
    def update_confidence(self, track_id: int, yolo_conf: float, 
                         color_score: float, shape_score: float, 
                         history: Dict) -> float:
        """
        Advanced adaptive confidence integration with context-based weighting
        
        Args:
            track_id: Track identifier
            yolo_conf: YOLO detector confidence
            color_score: Color similarity score
            shape_score: Shape similarity score
            history: Track history data
            
        Returns:
            float: Enhanced confidence score
        """
        # Step 1: Calculate ACF component
        acf_conf = self._calculate_acf(yolo_conf, color_score, shape_score, history)
        
        # Step 2: Calculate SHM component
        shm_conf = self._apply_shm(track_id, yolo_conf, history)
        
        # Step 3: Calculate HBM bonus
        hbm_bonus = self._calculate_history_bonus(history)
        
        # Step 4: Calculate track stability and maturity metrics
        stability = self._calculate_stability_factor(history)
        track_maturity = min(1.0, len(history.get('confidences', [])) / 10.0)
        occlusion = self._calculate_occlusion_factor(history)
        
        # Step 5: Calculate context-based ACF weight
        # - Reduce ACF influence for stable tracks
        # - Increase ACF influence during occlusion
        # - Gradually reduce ACF as track matures
        acf_weight = min(
            self.acf_max_weight,
            max(
                self.acf_min_weight,
                self.acf_max_weight - (stability * self.stability_impact) - 
                (track_maturity * self.maturity_impact) + (occlusion * 0.4)  # Ditingkatkan dari 0.3
            )
        )
        
        # Step 6: Combine ACF and SHM with context-based weighting
        combined_conf = (acf_weight * acf_conf) + ((1.0 - acf_weight) * shm_conf)
        
        # Step 7: Apply HBM bonus (always apply regardless of weighting)
        final_conf = min(1.0, combined_conf + hbm_bonus)
        
        # Debug logging for optimization
        if self.debug_level > 0 and track_id % 10 == 0:  # Log only every 10th track to reduce spam
            logger.info(f"Track {track_id}: acf={acf_conf:.3f}, shm={shm_conf:.3f}, " +
                       f"hbm_bonus={hbm_bonus:.3f}, stability={stability:.3f}, " +
                       f"acf_weight={acf_weight:.3f}, final={final_conf:.3f}")
        
        return final_conf
    
    def _calculate_acf(self, yolo_conf: float, color_score: float, 
                     shape_score: float, history: Dict) -> float:
        """Calculate confidence using Adaptive Confidence Fusion"""
        # Calculate stability and occlusion factors for ACF
        sfactor = self._calculate_stability_factor(history)
        ofactor = self._calculate_occlusion_factor(history)
        
        # Adjust weights for adaptive fusion
        weights = self._adjust_weights(sfactor, ofactor)
        
        # Calculate ACF confidence
        acf_conf = (weights['yolo'] * yolo_conf + 
                   weights['color'] * color_score + 
                   weights['shape'] * shape_score)
        
        return acf_conf
    
    def _apply_shm(self, track_id: int, yolo_conf: float, history: Dict) -> float:
        """Apply Smart History Management for temporal smoothing"""
        # Initialize history confidence if not exists
        if track_id not in self.history_confidence:
            self.history_confidence[track_id] = yolo_conf
            return yolo_conf
        
        # Calculate adaptive alpha
        nupdates = len(history.get('confidences', []))
        ofactor = self._calculate_occlusion_factor(history)
        
        # Modify adaptive alpha calculation
        αadaptive = min(self.αbase * (1 + nupdates/8) * (1 - ofactor * 0.8), self.αmax)
        
        # Apply temporal smoothing with adaptive alpha
        chist = self.history_confidence[track_id]
        cfinal = αadaptive * yolo_conf + (1 - αadaptive) * chist
        
        # Update history confidence
        self.history_confidence[track_id] = cfinal
        
        return cfinal
        
    def _calculate_stability_factor(self, history: Dict) -> float:
        """Calculate stability factor based on confidence and appearance history"""
        if not history or 'confidences' not in history:
            return 0.0
        
        # Calculate confidence stability
        σconf = np.std(history['confidences'][-5:]) if len(history['confidences']) >= 5 else 0.0
        
        # Calculate appearance stability if available
        if 'color_scores' in history and 'shape_scores' in history:
            color_std = np.std(history['color_scores'][-5:]) if len(history['color_scores']) >= 5 else 0.0
            shape_std = np.std(history['shape_scores'][-5:]) if len(history['shape_scores']) >= 5 else 0.0
            σappear = (color_std + shape_std) / 2
        else:
            σappear = 0.0
            
        # Modified calculation with less penalization for slight variation
        return exp(-1.8 * (σconf + σappear))  # Changed from -2.0 to -1.8
    
    def _calculate_occlusion_factor(self, history: Dict) -> float:
        """Calculate occlusion factor based on confidence changes with enhanced sensitivity"""
        if not history or 'confidences' not in history:
            return 0.0
            
        # Check if we have enough history
        if len(history.get('confidences', [])) < 3:
            return 0.0
            
        # Get recent confidence changes
        conf_changes = np.diff(history['confidences'][-3:])
        
        # Enhanced occlusion detection:
        # 1. Look for sudden drops in confidence
        min_change = min(conf_changes) if len(conf_changes) > 0 else 0.0
        
        # 2. Consider volatility in confidence as possible occlusion
        volatility = np.std(history['confidences'][-3:]) if len(history['confidences']) >= 3 else 0.0
        
        # Calculate occlusion factor with both drop and volatility
        occlusion_from_drop = abs(min_change) if min_change < -0.25 else 0.0  # Threshold lowered from -0.3
        occlusion_from_volatility = volatility * 0.5 if volatility > 0.15 else 0.0
        
        # Return combined occlusion factor (higher value indicates more likely occlusion)
        return max(occlusion_from_drop, occlusion_from_volatility)
        
    def _adjust_weights(self, sfactor: float, ofactor: float) -> Dict:
        """Adjust weights based on stability and occlusion factors"""
        weights = self.base_weights.copy()
        
        # Updated weight adjustment logic
        for key in weights:
            if key == 'yolo':
                # Reduce YOLO influence during occlusion
                weights[key] = weights[key] * (1 + sfactor * 0.8) * (1 - ofactor * 1.2)
            elif key == 'color':
                # Boost color during occlusion and for stable tracks
                weights[key] = weights[key] * (1 + sfactor * 1.1) * (1 + ofactor * 0.6)
            else:  # shape
                # Similar boost for shape
                weights[key] = weights[key] * (1 + sfactor * 1.1) * (1 + ofactor * 0.5)
            
        # Normalize weights
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
            
        return weights

    def _calculate_history_bonus(self, history: Dict) -> float:
        """Calculate history bonus based on confidence stability and history"""
        if not history or 'confidences' not in history:
            return 0.0
            
        confidences = history['confidences']
        if len(confidences) < 3:
            return 0.0
        
        # Calculate confidence stability
        σconf = np.std(confidences)
        
        # Calculate average historical confidence
        avg_conf = np.mean(confidences)
        
        # Enhanced bonus calculation:
        # - Give more weight to stability (2.2 instead of 2.4)
        # - More progressive bonus for tracks with high average confidence
        confidence_factor = avg_conf ** 0.8  # More reward for higher confidences
        
        # Calculate bonus with slightly more emphasis on stability
        hbonus = self.hmax * exp(-2.2 * σconf) * confidence_factor
        
        return hbonus
    
    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> int:
        """
        Membersihkan track yang sudah tidak aktif dari history_confidence
        
        Args:
            active_track_ids: List ID track yang masih aktif
            
        Returns:
            int: Jumlah track yang dibersihkan
        """
        # Simpan jumlah sebelum cleanup untuk menghitung berapa yang dihapus
        original_size = len(self.history_confidence)
        
        # Hapus track IDs yang tidak ada dalam active_track_ids
        stale_track_ids = [tid for tid in self.history_confidence.keys() 
                          if tid not in active_track_ids]
        
        for tid in stale_track_ids:
            del self.history_confidence[tid]
            
        # Hitung jumlah yang dihapus
        cleaned_count = original_size - len(self.history_confidence)
        
        return cleaned_count
    
    def update_track_ids(self, old_to_new_map: Dict[int, int]) -> None:
        """
        Update track IDs setelah ID reset
        
        Args:
            old_to_new_map: Mapping dari ID lama ke ID baru
        """
        # Buat dictionary baru dengan ID yang diperbarui
        updated_history = {}
        
        for old_id, conf in self.history_confidence.items():
            if old_id in old_to_new_map:
                new_id = old_to_new_map[old_id]
                updated_history[new_id] = conf
        
        # Ganti dictionary lama dengan yang baru
        self.history_confidence = updated_history