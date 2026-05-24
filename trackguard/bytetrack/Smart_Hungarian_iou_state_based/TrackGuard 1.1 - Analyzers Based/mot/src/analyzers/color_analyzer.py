import cv2
import numpy as np
from typing import Union, List, Dict, Tuple
from .base_analyzer import BaseAnalyzer

class ColorAnalyzer(BaseAnalyzer):
    """Analyze color characteristics of detected objects"""
    
    def __init__(self, 
                 hist_bins: int = 32,
                 hist_range: Tuple[int, int] = (0, 180)):
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        self.prev_hists = {}  # track_id -> previous histogram
        self.hist_history = {}  # track_id -> list of recent histograms
        self.max_hist_history = 5  # Store up to 5 recent histograms
        
    def analyze(self, frame: np.ndarray, bbox: Union[List, np.ndarray], 
                track_id: int = None) -> float:
        """
        Analyze color consistency in the region with occlusion robustness
        
        Args:
            frame: Current video frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Optional track ID for temporal analysis
            
        Returns:
            float: Color stability score between 0 and 1
        """
        # Extract ROI
        roi = self._extract_roi(frame, bbox)
        
        # Handle empty or invalid ROI
        if roi.size == 0 or roi.shape[0] <= 0 or roi.shape[1] <= 0:
            # If track_id exists and we have history, use history instead of failing
            if track_id is not None and track_id in self.prev_hists:
                return 0.95  # High confidence for historical data during occlusion
            return 0.0
            
        # Handle tiny ROIs that may indicate occlusion
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            # Likely occlusion - use historical data if available
            if track_id is not None and track_id in self.prev_hists:
                return 0.9  # High confidence for historical data during occlusion
            
        # Resize ROI for consistent analysis if needed
        # Only resize if ROI is big enough
        if roi.shape[0] > 1 and roi.shape[1] > 1:
            try:
                # Calculate new size with some safety checks
                target_size = (128, 128)  # Target size for consistent analysis
                
                # Calculate resize factors
                h_factor = target_size[0] / roi.shape[0]
                w_factor = target_size[1] / roi.shape[1]
                
                # Use the smaller factor to maintain aspect ratio
                resize_factor = min(h_factor, w_factor)
                
                # Calculate new dimensions
                new_h = max(1, int(roi.shape[0] * resize_factor))
                new_w = max(1, int(roi.shape[1] * resize_factor))
                
                # Only resize if both dimensions are valid (> 0)
                if new_w > 0 and new_h > 0:
                    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except cv2.error:
                # If resize fails, continue with original ROI
                pass
        
        # Convert to HSV
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error:
            # If conversion fails but we have history, use it instead of failing
            if track_id is not None and track_id in self.prev_hists:
                return 0.85  # Lower confidence for using history after conversion error
            return 0.0
        
        # Calculate current histogram
        try:
            current_hist = self._calculate_histogram(hsv_roi)
        except Exception:
            # If histogram calculation fails but we have history, use it
            if track_id is not None and track_id in self.prev_hists:
                return 0.85  # Lower confidence for using history after histogram error
            return 0.0
        
        # If no track_id or no previous histogram, just return high confidence
        if track_id is None or track_id not in self.prev_hists:
            if track_id is not None:
                self.prev_hists[track_id] = current_hist
                # Initialize history list
                self.hist_history[track_id] = [current_hist]
            return 1.0
            
        # Compare with previous histogram for normal case
        prev_hist = self.prev_hists[track_id]
        similarity = self._compare_histograms(current_hist, prev_hist)
        
        # If similarity is very low, it might be due to occlusion or lighting change
        # In that case, try comparing with several recent histograms to find best match
        if similarity < 0.5 and track_id in self.hist_history and len(self.hist_history[track_id]) > 1:
            # Try comparing with different historical histograms and take best match
            best_historical_similarity = max(
                self._compare_histograms(current_hist, hist)
                for hist in self.hist_history[track_id]
            )
            
            # If historical comparison is better, use a blend of current and historical
            if best_historical_similarity > similarity:
                similarity = similarity * 0.3 + best_historical_similarity * 0.7
        
        # Update previous histogram and history
        self.prev_hists[track_id] = current_hist
        
        # Update histogram history
        if track_id in self.hist_history:
            self.hist_history[track_id].append(current_hist)
            # Limit history size
            if len(self.hist_history[track_id]) > self.max_hist_history:
                self.hist_history[track_id] = self.hist_history[track_id][-self.max_hist_history:]
        else:
            self.hist_history[track_id] = [current_hist]
        
        return similarity
        
    def _calculate_histogram(self, hsv_image: np.ndarray) -> np.ndarray:
        """Calculate HSV histogram"""
        hist = cv2.calcHist([hsv_image], 
                           [0, 1],  # Use Hue and Saturation channels
                           None,
                           [self.hist_bins, self.hist_bins],
                           [self.hist_range[0], self.hist_range[1], 
                            0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
        
    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms using correlation"""
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, min(1, (similarity + 1) / 2))  # Scale from [-1,1] to [0,1]
    
    def analyze_color_change(self, frame: np.ndarray, 
                           bbox: Union[List, np.ndarray]) -> Dict:
        """
        Detailed analysis of color changes in region
        
        Args:
            frame: Current video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dict: Detailed color analysis results
        """
        roi = self._extract_roi(frame, bbox)
        if roi.size == 0:
            return {
                'dominant_colors': [],
                'color_variance': 0,
                'saturation_mean': 0
            }
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant colors
        pixels = hsv_roi.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_
        
        # Calculate color variance
        color_variance = np.var(pixels, axis=0).mean()
        
        # Calculate mean saturation
        saturation_mean = np.mean(pixels[:, 1])
        
        return {
            'dominant_colors': dominant_colors.tolist(),
            'color_variance': float(color_variance),
            'saturation_mean': float(saturation_mean)
        }
        
    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> int:
        """
        Remove histogram data for inactive tracks
        
        Args:
            active_track_ids: List of active track IDs
            
        Returns:
            int: Number of tracks cleaned up
        """
        # Get tracks to remove
        stale_tracks = set(self.prev_hists.keys()) - set(active_track_ids)
        
        # Clean up prev_hists
        for track_id in stale_tracks:
            if track_id in self.prev_hists:
                del self.prev_hists[track_id]
            if track_id in self.hist_history:
                del self.hist_history[track_id]
                
        return len(stale_tracks)
        
    def update_track_ids(self, old_to_new_map: Dict[int, int]) -> None:
        """
        Update track IDs after ID reset
        
        Args:
            old_to_new_map: Mapping from old to new IDs
        """
        # Update prev_hists
        new_prev_hists = {}
        for old_id, hist in self.prev_hists.items():
            if old_id in old_to_new_map:
                new_id = old_to_new_map[old_id]
                new_prev_hists[new_id] = hist
        self.prev_hists = new_prev_hists
        
        # Update hist_history
        new_hist_history = {}
        for old_id, hists in self.hist_history.items():
            if old_id in old_to_new_map:
                new_id = old_to_new_map[old_id]
                new_hist_history[new_id] = hists
        self.hist_history = new_hist_history