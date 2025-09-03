"""
Data preprocessing pipeline for brain tumor segmentation.
Includes skull stripping, normalization, and augmentation techniques.
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure
from skimage.transform import resize
from skimage.filters import gaussian
import tensorflow as tf


def skull_stripping(image, method='otsu'):
    """
    Perform skull stripping to remove non-brain tissue.
    
    Args:
        image: Input brain MRI image
        method: Method for skull stripping ('otsu', 'threshold', 'morphology')
        
    Returns:
        Skull-stripped image and brain mask
    """
    if len(image.shape) == 3:
        # Convert to grayscale if RGB
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    if method == 'otsu':
        # Otsu's thresholding
        threshold = filters.threshold_otsu(gray)
        brain_mask = gray > threshold
        
    elif method == 'threshold':
        # Simple thresholding
        _, brain_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        brain_mask = brain_mask.astype(bool)
        
    elif method == 'morphology':
        # Morphological operations
        threshold = filters.threshold_otsu(gray)
        brain_mask = gray > threshold
        
        # Remove small objects
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
        
        # Fill holes
        brain_mask = morphology.binary_fill_holes(brain_mask)
        
        # Morphological closing
        brain_mask = morphology.binary_closing(brain_mask, morphology.disk(5))
    
    # Apply mask to original image
    if len(image.shape) == 3:
        skull_stripped = image.copy()
        skull_stripped[~brain_mask] = 0
    else:
        skull_stripped = image.copy()
        skull_stripped[~brain_mask] = 0
    
    return skull_stripped, brain_mask


def gaussian_blur(image, sigma=1.0):
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        image: Input image
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred image
    """
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for i in range(image.shape[2]):
            blurred[:, :, i] = gaussian(image[:, :, i], sigma=sigma)
    else:
        blurred = gaussian(image, sigma=sigma)
    
    return blurred


def normalize_image(image, method='minmax'):
    """
    Normalize image intensity values.
    
    Args:
        image: Input image
        method: Normalization method ('minmax', 'zscore', 'percentile')
        
    Returns:
        Normalized image
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = image
            
    elif method == 'zscore':
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image
            
    elif method == 'percentile':
        # Percentile-based normalization
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            normalized = np.clip((image - p2) / (p98 - p2), 0, 1)
        else:
            normalized = image
    
    return normalized


def resize_image(image, target_size=(128, 128), preserve_range=True):
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        preserve_range: Whether to preserve the original value range
        
    Returns:
        Resized image
    """
    if len(image.shape) == 3:
        resized = resize(image, target_size + (image.shape[2],), 
                        preserve_range=preserve_range, anti_aliasing=True)
    else:
        resized = resize(image, target_size, 
                        preserve_range=preserve_range, anti_aliasing=True)
    
    return resized


def augment_image(image, mask=None, rotation_range=15, zoom_range=0.1, 
                 horizontal_flip=True, vertical_flip=False):
    """
    Apply data augmentation to image and corresponding mask.
    
    Args:
        image: Input image
        mask: Corresponding mask (optional)
        rotation_range: Range for random rotation in degrees
        zoom_range: Range for random zoom
        horizontal_flip: Whether to apply horizontal flip
        vertical_flip: Whether to apply vertical flip
        
    Returns:
        Augmented image and mask (if provided)
    """
    augmented_image = image.copy()
    augmented_mask = mask.copy() if mask is not None else None
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (w, h))
        if augmented_mask is not None:
            augmented_mask = cv2.warpAffine(augmented_mask, rotation_matrix, (w, h))
    
    # Random zoom
    if zoom_range > 0:
        zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize
        augmented_image = cv2.resize(augmented_image, (new_w, new_h))
        if augmented_mask is not None:
            augmented_mask = cv2.resize(augmented_mask, (new_w, new_h))
        
        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            augmented_image = augmented_image[start_h:start_h+h, start_w:start_w+w]
            if augmented_mask is not None:
                augmented_mask = augmented_mask[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            augmented_image = np.pad(augmented_image, 
                                   ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)) 
                                   if len(augmented_image.shape) == 3 else 
                                   ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)))
            if augmented_mask is not None:
                augmented_mask = np.pad(augmented_mask, 
                                      ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)))
    
    # Random horizontal flip
    if horizontal_flip and np.random.random() > 0.5:
        augmented_image = np.fliplr(augmented_image)
        if augmented_mask is not None:
            augmented_mask = np.fliplr(augmented_mask)
    
    # Random vertical flip
    if vertical_flip and np.random.random() > 0.5:
        augmented_image = np.flipud(augmented_image)
        if augmented_mask is not None:
            augmented_mask = np.flipud(augmented_mask)
    
    if augmented_mask is not None:
        return augmented_image, augmented_mask
    else:
        return augmented_image


def preprocess_pipeline(image, mask=None, target_size=(128, 128), 
                       apply_skull_stripping=True, apply_blur=True, 
                       apply_normalization=True, augmentation=False):
    """
    Complete preprocessing pipeline for brain tumor segmentation.
    
    Args:
        image: Input brain MRI image
        mask: Corresponding segmentation mask (optional)
        target_size: Target size for resizing
        apply_skull_stripping: Whether to apply skull stripping
        apply_blur: Whether to apply Gaussian blur
        apply_normalization: Whether to apply normalization
        augmentation: Whether to apply data augmentation
        
    Returns:
        Preprocessed image and mask (if provided)
    """
    processed_image = image.copy()
    processed_mask = mask.copy() if mask is not None else None
    
    # Skull stripping
    if apply_skull_stripping:
        processed_image, _ = skull_stripping(processed_image)
        if processed_mask is not None:
            processed_mask, _ = skull_stripping(processed_mask)
    
    # Gaussian blur
    if apply_blur:
        processed_image = gaussian_blur(processed_image, sigma=1.0)
    
    # Normalization
    if apply_normalization:
        processed_image = normalize_image(processed_image, method='minmax')
    
    # Resize
    processed_image = resize_image(processed_image, target_size)
    if processed_mask is not None:
        processed_mask = resize_image(processed_mask, target_size)
    
    # Data augmentation
    if augmentation:
        processed_image, processed_mask = augment_image(
            processed_image, processed_mask
        )
    
    if processed_mask is not None:
        return processed_image, processed_mask
    else:
        return processed_image


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    
    # Test preprocessing pipeline
    processed_image, processed_mask = preprocess_pipeline(
        dummy_image, dummy_mask, target_size=(128, 128)
    )
    
    print(f"Original image shape: {dummy_image.shape}")
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Original mask shape: {dummy_mask.shape}")
    print(f"Processed mask shape: {processed_mask.shape}")
    
    # Test individual functions
    skull_stripped, brain_mask = skull_stripping(dummy_image)
    blurred = gaussian_blur(dummy_image)
    normalized = normalize_image(dummy_image)
    
    print("Individual preprocessing steps completed successfully!")
