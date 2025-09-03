#!/usr/bin/env python3
"""
Example usage script for Brain Tumor Semantic Segmentation.

This script demonstrates how to use the implemented models, loss functions,
and evaluation metrics for brain tumor segmentation.

Usage:
    python example_usage.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append('src')

from models.unet import create_unet
from losses.focal_loss import FocalLoss
from utils.evaluation import calculate_metrics, dice_coefficient_numpy
from preprocessing.preprocessing import preprocess_pipeline


def create_sample_data():
    """Create sample data for demonstration."""
    # Create sample brain MRI image (simulated)
    np.random.seed(42)
    
    # Create a simple brain-like structure
    image = np.random.rand(128, 128, 3) * 0.3  # Background
    
    # Add brain tissue (brighter regions)
    center_x, center_y = 64, 64
    y, x = np.ogrid[:128, :128]
    brain_mask = (x - center_x)**2 + (y - center_y)**2 < 40**2
    image[brain_mask] += 0.4
    
    # Add tumor (even brighter, smaller region)
    tumor_mask = (x - center_x)**2 + (y - center_y)**2 < 15**2
    image[tumor_mask] += 0.3
    
    # Create corresponding mask
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[tumor_mask] = 1.0
    
    return image, mask


def demonstrate_preprocessing():
    """Demonstrate the preprocessing pipeline."""
    print("=== Preprocessing Demonstration ===")
    
    # Create sample data
    image, mask = create_sample_data()
    
    print(f"Original image shape: {image.shape}")
    print(f"Original mask shape: {mask.shape}")
    
    # Apply preprocessing
    processed_image, processed_mask = preprocess_pipeline(
        image, mask, target_size=(128, 128)
    )
    
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed mask shape: {processed_mask.shape}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Original Mask')
    axes[1].axis('off')
    
    axes[2].imshow(processed_image)
    axes[2].set_title('Processed Image')
    axes[2].axis('off')
    
    axes[3].imshow(processed_mask, cmap='gray')
    axes[3].set_title('Processed Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return processed_image, processed_mask


def demonstrate_model():
    """Demonstrate the U-Net model."""
    print("\n=== Model Demonstration ===")
    
    # Create model
    model = create_unet(input_shape=(128, 128, 3), filters=16)
    
    print("U-Net Model Summary:")
    model.summary()
    
    # Test with sample data
    sample_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    output = model(sample_input)
    
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
    
    return model


def demonstrate_loss_functions():
    """Demonstrate focal loss vs binary cross-entropy."""
    print("\n=== Loss Functions Demonstration ===")
    
    # Create sample predictions and ground truth
    y_true = np.random.randint(0, 2, (2, 128, 128, 1)).astype(np.float32)
    y_pred = np.random.rand(2, 128, 128, 1).astype(np.float32)
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss_value = focal_loss(y_true, y_pred)
    
    # Test binary cross-entropy
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_loss_value = tf.reduce_mean(bce_loss)
    
    print(f"Focal Loss: {focal_loss_value.numpy():.4f}")
    print(f"Binary Cross-Entropy: {bce_loss_value.numpy():.4f}")
    
    # Test with different alpha and gamma values
    print("\nFocal Loss with different parameters:")
    for alpha in [0.25, 0.5, 0.75]:
        for gamma in [1.0, 2.0, 3.0]:
            focal_loss_test = FocalLoss(alpha=alpha, gamma=gamma)
            loss_value = focal_loss_test(y_true, y_pred)
            print(f"  Alpha={alpha}, Gamma={gamma}: {loss_value.numpy():.4f}")


def demonstrate_evaluation():
    """Demonstrate evaluation metrics."""
    print("\n=== Evaluation Metrics Demonstration ===")
    
    # Create sample predictions and ground truth
    y_true = np.random.randint(0, 2, (128, 128)).astype(np.float32)
    y_pred = np.random.rand(128, 128).astype(np.float32)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Evaluation Metrics:")
    for metric_name, value in metrics.items():
        if not np.isinf(value):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")
    
    # Test Dice coefficient specifically
    dice = dice_coefficient_numpy(y_true, y_pred)
    print(f"\nDice Coefficient: {dice:.4f}")


def demonstrate_training_setup():
    """Demonstrate how to set up training."""
    print("\n=== Training Setup Demonstration ===")
    
    # Create model
    model = create_unet(input_shape=(128, 128, 3))
    
    # Create focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Compile model with focal loss
    model.compile(
        optimizer='adam',
        loss=focal_loss,
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("Model compiled with Focal Loss")
    print("Ready for training!")
    
    # Example of how to use with BCE loss
    model_bce = create_unet(input_shape=(128, 128, 3))
    model_bce.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("Model compiled with Binary Cross-Entropy Loss")
    print("Ready for training!")


def main():
    """Main demonstration function."""
    print("Brain Tumor Semantic Segmentation - Example Usage")
    print("=" * 50)
    
    try:
        # Demonstrate each component
        demonstrate_preprocessing()
        demonstrate_model()
        demonstrate_loss_functions()
        demonstrate_evaluation()
        demonstrate_training_setup()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nTo use this code with your own data:")
        print("1. Place your images in data/images/")
        print("2. Place your masks in data/masks/")
        print("3. Run the training_comparison.ipynb notebook")
        print("4. Or use the individual modules in your own scripts")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
