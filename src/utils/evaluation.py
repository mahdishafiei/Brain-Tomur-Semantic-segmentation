"""
Evaluation metrics for brain tumor segmentation.
Includes Dice coefficient, Hausdorff distance, and other segmentation metrics.
"""

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score, accuracy_score
import cv2


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient (F1-score) for segmentation.
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_coefficient_numpy(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient using numpy.
    
    Args:
        y_true: Ground truth binary mask (numpy array)
        y_pred: Predicted binary mask (numpy array)
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient value
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def hausdorff_distance(y_true, y_pred):
    """
    Calculate Hausdorff distance between two binary masks.
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        
    Returns:
        Hausdorff distance value
    """
    # Convert to binary masks
    y_true = (y_true > 0.5).astype(np.uint8)
    y_pred = (y_pred > 0.5).astype(np.uint8)
    
    # Find contours
    contours_true, _ = cv2.findContours(y_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv2.findContours(y_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_true) == 0 or len(contours_pred) == 0:
        return float('inf')
    
    # Get the largest contour for each mask
    contour_true = max(contours_true, key=cv2.contourArea)
    contour_pred = max(contours_pred, key=cv2.contourArea)
    
    # Reshape contours for hausdorff distance calculation
    points_true = contour_true.reshape(-1, 2)
    points_pred = contour_pred.reshape(-1, 2)
    
    # Calculate directed Hausdorff distances
    d1 = directed_hausdorff(points_true, points_pred)[0]
    d2 = directed_hausdorff(points_pred, points_true)[0]
    
    # Return the maximum (symmetric Hausdorff distance)
    return max(d1, d2)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate comprehensive evaluation metrics for segmentation.
    
    Args:
        y_true: Ground truth masks (numpy array)
        y_pred: Predicted masks (numpy array)
        threshold: Threshold for binary conversion
        
    Returns:
        Dictionary containing all metrics
    """
    # Convert to binary masks
    y_true_binary = (y_true > threshold).astype(np.uint8)
    y_pred_binary = (y_pred > threshold).astype(np.uint8)
    
    # Flatten for sklearn metrics
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Calculate metrics
    metrics = {}
    
    # Pixel-wise metrics
    metrics['accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
    metrics['precision'] = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    metrics['recall'] = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Dice coefficient
    metrics['dice'] = dice_coefficient_numpy(y_true, y_pred)
    
    # Hausdorff distance
    try:
        metrics['hausdorff'] = hausdorff_distance(y_true, y_pred)
    except:
        metrics['hausdorff'] = float('inf')
    
    # IoU (Intersection over Union)
    intersection = np.sum(y_true_binary * y_pred_binary)
    union = np.sum(y_true_binary) + np.sum(y_pred_binary) - intersection
    metrics['iou'] = intersection / (union + 1e-6)
    
    # Sensitivity and Specificity
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    metrics['sensitivity'] = tp / (tp + fn + 1e-6)  # Recall
    metrics['specificity'] = tn / (tn + fp + 1e-6)
    
    return metrics


def calculate_metrics_batch(y_true_batch, y_pred_batch, threshold=0.5):
    """
    Calculate metrics for a batch of images.
    
    Args:
        y_true_batch: Batch of ground truth masks
        y_pred_batch: Batch of predicted masks
        threshold: Threshold for binary conversion
        
    Returns:
        Dictionary with mean and std of all metrics
    """
    all_metrics = []
    
    for i in range(len(y_true_batch)):
        metrics = calculate_metrics(y_true_batch[i], y_pred_batch[i], threshold)
        all_metrics.append(metrics)
    
    # Calculate mean and standard deviation
    result = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isinf(m[key])]
        if values:
            result[f'{key}_mean'] = np.mean(values)
            result[f'{key}_std'] = np.std(values)
        else:
            result[f'{key}_mean'] = 0.0
            result[f'{key}_std'] = 0.0
    
    return result


class SegmentationMetrics(tf.keras.metrics.Metric):
    """
    Custom Keras metric for Dice coefficient.
    """
    
    def __init__(self, name='dice_coefficient', **kwargs):
        super(SegmentationMetrics, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        dice = dice_coefficient(y_true, y_pred)
        self.dice.assign_add(dice)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.dice / self.count
    
    def reset_state(self):
        self.dice.assign(0.0)
        self.count.assign(0.0)


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    y_true = np.random.randint(0, 2, (128, 128)).astype(np.float32)
    y_pred = np.random.rand(128, 128).astype(np.float32)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Test batch metrics
    y_true_batch = np.random.randint(0, 2, (5, 128, 128)).astype(np.float32)
    y_pred_batch = np.random.rand(5, 128, 128).astype(np.float32)
    
    batch_metrics = calculate_metrics_batch(y_true_batch, y_pred_batch)
    
    print("\nBatch Metrics:")
    for key, value in batch_metrics.items():
        print(f"{key}: {value:.4f}")
