import cv2
import numpy as np
from typing import Union, List, Dict
from .base_analyzer import BaseAnalyzer

class ShapeAnalyzer(BaseAnalyzer):
    """Analyze shape characteristics of detected objects"""
    
    def __init__(self):
        self.prev_contours = {}  # track_id -> previous contours
        self.contour_history = {}  # track_id -> list of recent contours metrics
        self.max_contour_history = 5  # Store metrics for up to 5 recent frames
        self.frame_id = 0  # Current frame ID for contour caching
        
    def set_frame_id(self, frame_id: int):
        """Set current frame ID for contour caching"""
        self.frame_id = frame_id
        
    def analyze(self, frame: np.ndarray, bbox: Union[List, np.ndarray], 
                track_id: int = None) -> float:
        """
        Analyze shape stability in the region with occlusion robustness
        
        Args:
            frame: Current video frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Optional track ID for temporal analysis
            
        Returns:
            float: Shape stability score between 0 and 1
        """
        # Extract ROI
        roi = self._extract_roi(frame, bbox)
        
        # Handle empty or invalid ROI
        if roi.size == 0 or roi.shape[0] <= 0 or roi.shape[1] <= 0:
            # If track exists and we have history, use history instead of failing
            if track_id is not None and track_id in self.prev_contours:
                return 0.95  # High confidence for historical data during occlusion
            return 0.0
        
        # Handle tiny ROIs that may indicate occlusion
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            # Likely occlusion - use historical data if available
            if track_id is not None and track_id in self.prev_contours:
                return 0.9  # High confidence for historical data during occlusion
            
        # Get current contours
        try:
            current_contours = self._get_contours(roi)
        except cv2.error:
            # If contour extraction fails but we have history, use it
            if track_id is not None and track_id in self.prev_contours:
                return 0.85  # Lower confidence for using history after error
            return 0.0
        
        # If no track_id or no previous contours, return high confidence
        if track_id is None or track_id not in self.prev_contours:
            if track_id is not None:
                self.prev_contours[track_id] = current_contours
                # Store contour metrics for history
                self._update_contour_history(track_id, current_contours)
            return 1.0
            
        # Compare with previous contours
        prev_contours = self.prev_contours[track_id]
        similarity = self._compare_contours(current_contours, prev_contours)
        
        # If similarity is very low, it might be due to occlusion
        # Try comparing with historical contour metrics
        if similarity < 0.5 and track_id in self.contour_history and len(self.contour_history[track_id]) > 1:
            # Get current contour metrics
            current_metrics = self._calculate_contour_metrics(current_contours)
            
            # Find best matching historical contour metrics
            best_historical_similarity = 0
            for historical_metrics in self.contour_history[track_id]:
                hist_similarity = self._compare_contour_metrics(current_metrics, historical_metrics)
                best_historical_similarity = max(best_historical_similarity, hist_similarity)
            
            # If historical comparison is better, use a blend
            if best_historical_similarity > similarity:
                similarity = similarity * 0.3 + best_historical_similarity * 0.7
        
        # Update previous contours
        self.prev_contours[track_id] = current_contours
        
        # Update contour history
        self._update_contour_history(track_id, current_contours)
        
        return similarity
        
    def _update_contour_history(self, track_id: int, contours: List):
        """Update contour metrics history for a track"""
        # Calculate metrics for current contours
        metrics = self._calculate_contour_metrics(contours)
        
        # Add to history
        if track_id not in self.contour_history:
            self.contour_history[track_id] = [metrics]
        else:
            self.contour_history[track_id].append(metrics)
            # Limit history size
            if len(self.contour_history[track_id]) > self.max_contour_history:
                self.contour_history[track_id] = self.contour_history[track_id][-self.max_contour_history:]
        
    def _calculate_contour_metrics(self, contours: List) -> Dict:
        """Calculate key metrics for contours"""
        if not contours:
            return {
                'count': 0,
                'total_area': 0,
                'total_perimeter': 0,
                'avg_complexity': 0
            }
            
        try:
            # Calculate areas and perimeters
            areas = [cv2.contourArea(cnt) for cnt in contours if cnt.size > 0]
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours if cnt.size > 0]
            
            # Handle empty lists
            if not areas or not perimeters:
                return {
                    'count': 0,
                    'total_area': 0,
                    'total_perimeter': 0,
                    'avg_complexity': 0
                }
            
            # Calculate shape complexity (perimeter^2 / area)
            complexities = [(p**2 / (a + 1e-6)) for p, a in zip(perimeters, areas)]
            
            return {
                'count': len(contours),
                'total_area': sum(areas),
                'total_perimeter': sum(perimeters),
                'avg_complexity': sum(complexities) / len(complexities) if complexities else 0
            }
        except:
            # If calculation fails, return empty metrics
            return {
                'count': len(contours),
                'total_area': 0,
                'total_perimeter': 0,
                'avg_complexity': 0
            }
            
    def _compare_contour_metrics(self, metrics1: Dict, metrics2: Dict) -> float:
        """Compare two sets of contour metrics"""
        if not metrics1 or not metrics2:
            return 0.0
            
        # Compare count
        count_similarity = min(metrics1['count'], metrics2['count']) / (max(metrics1['count'], metrics2['count']) + 1e-6)
        
        # Compare area
        area1 = metrics1['total_area']
        area2 = metrics2['total_area']
        area_similarity = min(area1, area2) / (max(area1, area2) + 1e-6) if area1 > 0 and area2 > 0 else 0
        
        # Compare complexity
        complexity1 = metrics1['avg_complexity']
        complexity2 = metrics2['avg_complexity']
        complexity_similarity = 1.0 - min(1.0, abs(complexity1 - complexity2) / (max(complexity1, complexity2) + 1e-6))
        
        # Weighted combination
        return 0.4 * count_similarity + 0.4 * area_similarity + 0.2 * complexity_similarity
        
    def _get_contours(self, roi: np.ndarray) -> List:
        """Extract contours from ROI"""
        # Check if ROI is valid
        if roi.size == 0 or roi.shape[0] <= 0 or roi.shape[1] <= 0:
            return []
            
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
                
        # Convert to grayscale
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # If conversion fails, return empty contours
            return []
        
        # Apply edge detection with additional noise filtering
        try:
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges slightly for more robust contour detection
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        except cv2.error:
            # If edge detection fails, return empty contours
            return []
        
        # Find contours
        try:
            contours, _ = cv2.findContours(edges, 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        except cv2.error:
            # If contour finding fails, return empty contours
            return []
        
        # Filter out tiny or invalid contours
        valid_contours = [cnt for cnt in contours if cnt.size >= 5]
        
        return valid_contours
        
    def _compare_contours(self, contours1: List, contours2: List) -> float:
        """Compare two sets of contours"""
        if not contours1 or not contours2:
            return 0.0
        
        # Handle case where contours might be empty lists
        if len(contours1) == 0 or len(contours2) == 0:
            return 0.0
            
        # Calculate total contour areas
        try:
            area1 = sum(cv2.contourArea(cnt) for cnt in contours1 if cnt.size > 0)
            area2 = sum(cv2.contourArea(cnt) for cnt in contours2 if cnt.size > 0)
        except:
            # If area calculation fails, return default value
            return 0.0
        
        # Avoid division by zero
        if area1 <= 0 or area2 <= 0:
            return 0.0
        
        # Calculate area similarity - the main metric
        area_similarity = min(area1, area2) / (max(area1, area2) + 1e-6)
        
        # Calculate count similarity - secondary metric
        count_similarity = min(len(contours1), len(contours2)) / (max(len(contours1), len(contours2)) + 1e-6)
        
        # Calculate perimeter similarity if possible
        try:
            perimeter1 = sum(cv2.arcLength(cnt, True) for cnt in contours1 if cnt.size > 0)
            perimeter2 = sum(cv2.arcLength(cnt, True) for cnt in contours2 if cnt.size > 0)
            
            if perimeter1 > 0 and perimeter2 > 0:
                perimeter_similarity = min(perimeter1, perimeter2) / (max(perimeter1, perimeter2) + 1e-6)
            else:
                perimeter_similarity = 0
        except:
            perimeter_similarity = 0
        
        # Weighted combination of metrics - area is most important
        return 0.6 * area_similarity + 0.3 * count_similarity + 0.1 * perimeter_similarity
    
    def analyze_shape_features(self, frame: np.ndarray, 
                             bbox: Union[List, np.ndarray]) -> Dict:
        """
        Detailed analysis of shape features
        
        Args:
            frame: Current video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dict: Detailed shape analysis results
        """
        roi = self._extract_roi(frame, bbox)
        if roi.size == 0:
            return {
                'contour_count': 0,
                'area': 0,
                'perimeter': 0,
                'complexity': 0
            }
            
        contours = self._get_contours(roi)
        
        if not contours:
            return {
                'contour_count': 0,
                'area': 0,
                'perimeter': 0,
                'complexity': 0
            }
            
        # Calculate shape metrics
        try:
            areas = [cv2.contourArea(cnt) for cnt in contours if cnt.size > 0]
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours if cnt.size > 0]
            
            if not areas or not perimeters:
                return {
                    'contour_count': len(contours),
                    'area': 0,
                    'perimeter': 0,
                    'complexity': 0
                }
            
            # Calculate shape complexity (perimeter^2 / area)
            complexities = [(p**2 / (a + 1e-6)) for p, a in zip(perimeters, areas)]
            
            return {
                'contour_count': len(contours),
                'area': sum(areas),
                'perimeter': sum(perimeters),
                'complexity': sum(complexities) / len(complexities) if complexities else 0
            }
        except:
            # If metrics calculation fails, return default values
            return {
                'contour_count': len(contours),
                'area': 0,
                'perimeter': 0,
                'complexity': 0
            }
            
    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> int:
        """
        Remove contour data for inactive tracks
        
        Args:
            active_track_ids: List of active track IDs
            
        Returns:
            int: Number of tracks cleaned up
        """
        # Get tracks to remove
        stale_tracks = set(self.prev_contours.keys()) - set(active_track_ids)
        
        # Clean up prev_contours and contour_history
        for track_id in stale_tracks:
            if track_id in self.prev_contours:
                del self.prev_contours[track_id]
            if track_id in self.contour_history:
                del self.contour_history[track_id]
                
        return len(stale_tracks)
        
    def update_track_ids(self, old_to_new_map: Dict[int, int]) -> None:
        """
        Update track IDs after ID reset
        
        Args:
            old_to_new_map: Mapping from old to new IDs
        """
        # Update prev_contours
        new_prev_contours = {}
        for old_id, contours in self.prev_contours.items():
            if old_id in old_to_new_map:
                new_id = old_to_new_map[old_id]
                new_prev_contours[new_id] = contours
        self.prev_contours = new_prev_contours
        
        # Update contour_history
        new_contour_history = {}
        for old_id, history in self.contour_history.items():
            if old_id in old_to_new_map:
                new_id = old_to_new_map[old_id]
                new_contour_history[new_id] = history
        self.contour_history = new_contour_history