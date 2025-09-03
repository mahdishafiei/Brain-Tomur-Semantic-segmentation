"""
Focal Loss implementation for brain tumor segmentation.
Based on the paper: "Focal Loss for Dense Object Detection" by Lin et al.
"""

import tensorflow as tf
import tensorflow.keras.backend as K


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss implementation for addressing class imbalance in segmentation tasks.
    
    The focal loss is designed to down-weight easy examples and focus on hard examples
    by using a modulating factor (1-p_t)^gamma.
    
    Args:
        alpha (float): Weighting factor for rare class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        from_logits (bool): Whether predictions are logits or probabilities
        reduction (str): Type of reduction to apply to loss
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, reduction='auto', name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """
        Compute focal loss.
        
        Args:
            y_true: Ground truth labels (binary mask)
            y_pred: Predicted probabilities or logits
            
        Returns:
            Focal loss value
        """
        if self.from_logits:
            # Convert logits to probabilities
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Ensure y_pred is in valid range [0, 1]
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Calculate focal weight
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight * ce
        
        return focal_loss
    
    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


def focal_loss_fixed(alpha=0.25, gamma=2.0, from_logits=False):
    """
    Alternative focal loss implementation as a function.
    
    Args:
        alpha (float): Weighting factor for rare class
        gamma (float): Focusing parameter
        from_logits (bool): Whether predictions are logits or probabilities
        
    Returns:
        Focal loss function
    """
    def focal_loss_fn(y_true, y_pred):
        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Calculate focal weight
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight * ce
        
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fn


# Example usage and testing
if __name__ == "__main__":
    # Test the focal loss implementation
    import numpy as np
    
    # Create sample data
    y_true = tf.constant([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=tf.float32)
    y_pred = tf.constant([[[0.9, 0.1], [0.2, 0.8]], [[0.1, 0.9], [0.8, 0.2]]], dtype=tf.float32)
    
    # Test class-based implementation
    focal_loss_class = FocalLoss(alpha=0.25, gamma=2.0)
    loss_class = focal_loss_class(y_true, y_pred)
    print(f"Focal Loss (class): {loss_class.numpy()}")
    
    # Test function-based implementation
    focal_loss_fn = focal_loss_fixed(alpha=0.25, gamma=2.0)
    loss_fn = focal_loss_fn(y_true, y_pred)
    print(f"Focal Loss (function): {loss_fn.numpy()}")
    
    # Compare with binary cross-entropy
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    print(f"Binary Cross-Entropy: {tf.reduce_mean(bce_loss).numpy()}")
