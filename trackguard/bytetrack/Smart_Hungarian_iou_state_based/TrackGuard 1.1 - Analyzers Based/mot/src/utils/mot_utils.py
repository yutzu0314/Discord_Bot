import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import configparser
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

def parse_seq_info(seq_path: Union[str, Path]) -> Dict:
    """
    Parse sequence information from seqinfo.ini
    
    Args:
        seq_path: Path to sequence directory
        
    Returns:
        Dict: Sequence information
    """
    seq_path = Path(seq_path)
    seqinfo_path = seq_path / 'seqinfo.ini'
    
    if not seqinfo_path.exists():
        # Create default values if file doesn't exist
        return {
            'name': seq_path.name,
            'imDir': 'img1',
            'frameRate': 30,
            'seqLength': len(list((seq_path / 'img1').glob('*.jpg'))),
            'imWidth': 1920,
            'imHeight': 1080,
            'imExt': '.jpg'
        }
    
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    
    return {
        'name': config['Sequence']['name'],
        'imDir': config['Sequence']['imDir'],
        'frameRate': int(config['Sequence']['frameRate']),
        'seqLength': int(config['Sequence']['seqLength']),
        'imWidth': int(config['Sequence']['imWidth']),
        'imHeight': int(config['Sequence']['imHeight']),
        'imExt': config['Sequence']['imExt']
    }

def read_mot_ground_truth(gt_file_path: Union[str, Path]) -> Dict[int, List[Dict]]:
    """
    Read MOT Challenge ground truth file
    
    Args:
        gt_file_path: Path to gt.txt file
        
    Returns:
        Dict: Frame index -> List of ground truth objects
    """
    ground_truth = defaultdict(list)
    
    gt_file_path = Path(gt_file_path)
    if not gt_file_path.exists():
        print(f"Warning: Ground truth file not found: {gt_file_path}")
        return ground_truth
    
    with open(gt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:  # At minimum need frame,id,x,y,w,h
                continue
                
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            class_id = int(parts[7]) if len(parts) > 7 else 1  # Default to class 1 (often "person")
            
            # Convert to [x1,y1,x2,y2] format for easier IoU calculation
            bbox = [x, y, x+w, y+h]
            
            ground_truth[frame_idx].append({
                'track_id': track_id,
                'bbox': bbox,
                'confidence': conf,
                'class_id': class_id
            })
    
    return ground_truth

def get_frame_gt(gt_data: Dict[int, List[Dict]], frame_idx: int) -> List[Dict]:
    """
    Get ground truth data for a specific frame
    
    Args:
        gt_data: Ground truth data from read_mot_ground_truth
        frame_idx: Frame index
        
    Returns:
        List[Dict]: Ground truth objects for the frame
    """
    return gt_data.get(frame_idx, [])

def export_tracking_results(results: List[List], output_file: Union[str, Path]) -> None:
    """
    Export tracking results in MOT Challenge format
    
    Args:
        results: List of tracking results [frame, id, x, y, w, h, conf, class, -1]
        output_file: Path to output file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)

def visualize_tracking_results(frame: np.ndarray, 
                             pred_objects: List[Dict], 
                             gt_objects: Optional[List[Dict]] = None,
                             frame_metrics: Optional[Dict] = None) -> np.ndarray:
    """
    Visualize tracking results with optional ground truth and metrics
    
    Args:
        frame: Input frame
        pred_objects: List of predicted objects
        gt_objects: Optional list of ground truth objects
        frame_metrics: Optional metrics for the current frame
        
    Returns:
        np.ndarray: Visualization image
    """
    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]
    
    # Draw ground truth if available
    if gt_objects:
        for gt in gt_objects:
            x1, y1, x2, y2 = map(int, gt['bbox'])
            track_id = gt['track_id']
            
            # Draw bounding box in green
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID
            cv2.putText(vis_frame, f"GT:{track_id}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw predictions
    for pred in pred_objects:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        track_id = pred['track_id']
        conf = pred['confidence']
        class_id = pred.get('class_id', 0)
        
        # Generate color based on track_id
        color = (int(hash(str(track_id)) % 255), 
                int(hash(str(track_id*2)) % 255),
                int(hash(str(track_id*3)) % 255))
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID and confidence
        label = f"ID:{track_id} C:{class_id} {conf:.2f}"
        cv2.putText(vis_frame, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw metrics if available
    if frame_metrics:
        metrics_text = [
            f"MOTA: {frame_metrics.get('mota', 0):.2f}",
            f"Precision: {frame_metrics.get('precision', 0):.2f}",
            f"Recall: {frame_metrics.get('recall', 0):.2f}",
            f"TP: {frame_metrics.get('tp', 0)}",
            f"FP: {frame_metrics.get('fp', 0)}",
            f"FN: {frame_metrics.get('fn', 0)}",
            f"ID Sw: {frame_metrics.get('id_switches', 0)}"
        ]
        
        y_offset = 30
        for text in metrics_text:
            cv2.putText(vis_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
    
    return vis_frame

def plot_mot_metrics(metrics_history: List[Dict], output_file: Union[str, Path] = None) -> plt.Figure:
    """
    Create visualization plots for MOT metrics
    
    Args:
        metrics_history: List of metrics dictionaries
        output_file: Optional path to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    if not metrics_history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "No metrics data available", 
               ha='center', va='center', fontsize=14)
        return fig
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract metrics over time
    frames = list(range(1, len(metrics_history) + 1))
    mota = [m.get('mota', 0) for m in metrics_history]
    precision = [m.get('precision', 0) for m in metrics_history]
    recall = [m.get('recall', 0) for m in metrics_history]
    fp = [m.get('fp', 0) for m in metrics_history]
    fn = [m.get('fn', 0) for m in metrics_history]
    id_switches = [m.get('id_switches', 0) for m in metrics_history]
    
    # Plot MOTA
    axs[0, 0].plot(frames, mota, 'b-', label='MOTA')
    axs[0, 0].set_title('MOTA over Time')
    axs[0, 0].set_xlabel('Frame')
    axs[0, 0].set_ylabel('MOTA')
    axs[0, 0].grid(True)
    
    # Plot Precision and Recall
    axs[0, 1].plot(frames, precision, 'g-', label='Precision')
    axs[0, 1].plot(frames, recall, 'r-', label='Recall')
    axs[0, 1].set_title('Precision and Recall')
    axs[0, 1].set_xlabel('Frame')
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot False Positives and False Negatives
    axs[1, 0].plot(frames, fp, 'r-', label='False Positives')
    axs[1, 0].plot(frames, fn, 'b-', label='False Negatives')
    axs[1, 0].set_title('FP and FN over Time')
    axs[1, 0].set_xlabel('Frame')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot ID Switches
    axs[1, 1].plot(frames, id_switches, 'r-', label='ID Switches')
    axs[1, 1].set_title('ID Switches over Time')
    axs[1, 1].set_xlabel('Frame')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    
    return fig

def plot_mot_summary(summary_metrics: Dict, output_file: Union[str, Path] = None) -> plt.Figure:
    """
    Create visualization of MOT metrics summary
    
    Args:
        summary_metrics: Summary metrics from MOTMetrics.get_summary()
        output_file: Optional path to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Upper left: Main metrics
    metrics = ['mota', 'motp', 'idf1']
    values = [summary_metrics.get(m, 0) for m in metrics]
    axs[0, 0].bar(metrics, values, color=['blue', 'green', 'red'])
    axs[0, 0].set_title('Primary MOT Metrics')
    axs[0, 0].set_ylim(0, 1)
    for i, v in enumerate(values):
        axs[0, 0].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Upper right: Tracking ratios
    ratio_metrics = ['mt_ratio', 'pt_ratio', 'ml_ratio']
    ratio_labels = ['Mostly Tracked', 'Partly Tracked', 'Mostly Lost']
    ratio_values = [summary_metrics.get(m, 0) for m in ratio_metrics]
    axs[0, 1].pie(ratio_values, labels=ratio_labels, autopct='%1.1f%%', 
                 colors=['green', 'orange', 'red'])
    axs[0, 1].set_title('Tracking Quality Distribution')
    
    # Lower left: Precision-Recall
    precision = summary_metrics.get('precision', 0)
    recall = summary_metrics.get('recall', 0)
    axs[1, 0].scatter(recall, precision, s=100, c='blue')
    axs[1, 0].set_xlim(0, 1)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_xlabel('Recall')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].set_title('Precision-Recall')
    axs[1, 0].grid(True)
    axs[1, 0].text(recall, precision - 0.05, f"({recall:.3f}, {precision:.3f})", ha='center')
    
    # Lower right: ID Metrics
    id_metrics = ['id_switches', 'fragmentations']
    id_values = [summary_metrics.get(m, 0) for m in id_metrics]
    axs[1, 1].bar(id_metrics, id_values, color=['red', 'orange'])
    axs[1, 1].set_title('ID Metrics')
    for i, v in enumerate(id_values):
        axs[1, 1].text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    
    return fig

def create_mot_folder_structure(output_path: Union[str, Path], 
                             seq_name: str = "SEQ01") -> Path:
    """
    Create MOT Challenge folder structure
    
    Args:
        output_path: Base output path
        seq_name: Sequence name
        
    Returns:
        Path: Path to created sequence folder
    """
    output_path = Path(output_path)
    
    # Create train sequence folder
    train_seq_path = output_path / 'train' / seq_name
    train_seq_path.mkdir(parents=True, exist_ok=True)
    
    # Create img1 folder
    img_folder = train_seq_path / 'img1'
    img_folder.mkdir(exist_ok=True)
    
    # Create gt folder
    gt_folder = train_seq_path / 'gt'
    gt_folder.mkdir(exist_ok=True)
    
    return train_seq_path

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU score
    """
    # Calculate intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def create_seqinfo_file(seq_path: Union[str, Path],
                      seq_name: str,
                      frame_rate: int,
                      seq_length: int,
                      im_width: int,
                      im_height: int,
                      im_ext: str = '.jpg') -> None:
    """
    Create seqinfo.ini file for MOT sequence
    
    Args:
        seq_path: Path to sequence folder
        seq_name: Sequence name
        frame_rate: Frame rate
        seq_length: Number of frames
        im_width: Image width
        im_height: Image height
        im_ext: Image extension
    """
    config = configparser.ConfigParser()
    config['Sequence'] = {
        'name': seq_name,
        'imDir': 'img1',
        'frameRate': str(frame_rate),
        'seqLength': str(seq_length),
        'imWidth': str(im_width),
        'imHeight': str(im_height),
        'imExt': im_ext
    }
    
    with open(Path(seq_path) / 'seqinfo.ini', 'w') as f:
        config.write(f)

def convert_to_mot_format(bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert [x1, y1, x2, y2] format to MOT format [x, y, width, height]
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        
    Returns:
        Tuple: (x, y, width, height)
    """
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    return x, y, width, height