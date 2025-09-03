"""
U-Net model implementation for brain tumor segmentation.
Based on the original U-Net paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class UNet(keras.Model):
    """
    U-Net architecture for semantic segmentation.
    
    The U-Net consists of a contracting path (encoder) and an expansive path (decoder).
    Skip connections between corresponding layers help preserve fine-grained details.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        num_classes (int): Number of output classes (default: 1 for binary segmentation)
        filters (int): Number of filters in the first layer (default: 16)
        dropout_rate (float): Dropout rate for regularization (default: 0.1)
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=1, filters=16, dropout_rate=0.1, **kwargs):
        super(UNet, self).__init__(**kwargs)
        
        self.input_shape_val = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.dropout_rate = dropout_rate
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the U-Net architecture."""
        inputs = layers.Input(shape=self.input_shape_val, name='input')
        
        # Normalize inputs
        s = layers.Lambda(lambda x: x / 255.0, name='normalize')(inputs)
        
        # Contracting path (Encoder)
        c1, p1 = self._conv_block(s, self.filters, 'c1')
        c2, p2 = self._conv_block(p1, self.filters * 2, 'c2')
        c3, p3 = self._conv_block(p2, self.filters * 4, 'c3')
        c4, p4 = self._conv_block(p3, self.filters * 8, 'c4')
        
        # Bottleneck
        c5 = self._conv_block(p4, self.filters * 16, 'c5', pooling=False)
        
        # Expansive path (Decoder)
        u6 = self._upconv_block(c5, c4, self.filters * 8, 'u6')
        u7 = self._upconv_block(u6, c3, self.filters * 4, 'u7')
        u8 = self._upconv_block(u7, c2, self.filters * 2, 'u8')
        u9 = self._upconv_block(u8, c1, self.filters, 'u9')
        
        # Output layer
        outputs = layers.Conv2D(
            self.num_classes, 
            (1, 1), 
            activation='sigmoid', 
            name='output'
        )(u9)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='unet')
    
    def _conv_block(self, x, filters, name, pooling=True):
        """
        Convolutional block with two conv layers, dropout, and optional pooling.
        
        Args:
            x: Input tensor
            filters: Number of filters
            name: Block name prefix
            pooling: Whether to include max pooling
            
        Returns:
            Tuple of (conv_output, pool_output) if pooling=True, else conv_output
        """
        # First convolution
        c = layers.Conv2D(
            filters, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same',
            name=f'{name}_conv1'
        )(x)
        c = layers.Dropout(self.dropout_rate, name=f'{name}_dropout1')(c)
        
        # Second convolution
        c = layers.Conv2D(
            filters, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same',
            name=f'{name}_conv2'
        )(c)
        c = layers.Dropout(self.dropout_rate, name=f'{name}_dropout2')(c)
        
        if pooling:
            p = layers.MaxPooling2D((2, 2), name=f'{name}_pool')(c)
            return c, p
        else:
            return c
    
    def _upconv_block(self, x, skip, filters, name):
        """
        Upconvolutional block with skip connection.
        
        Args:
            x: Input tensor from previous layer
            skip: Skip connection from encoder
            filters: Number of filters
            name: Block name prefix
            
        Returns:
            Upconvolutional block output
        """
        # Upconvolution
        u = layers.Conv2DTranspose(
            filters, (2, 2), 
            strides=(2, 2), 
            padding='same',
            name=f'{name}_upconv'
        )(x)
        
        # Concatenate with skip connection
        u = layers.concatenate([u, skip], name=f'{name}_concat')
        
        # Two convolutions
        u = layers.Conv2D(
            filters, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same',
            name=f'{name}_conv1'
        )(u)
        u = layers.Dropout(self.dropout_rate, name=f'{name}_dropout1')(u)
        
        u = layers.Conv2D(
            filters, (3, 3), 
            activation='relu', 
            kernel_initializer='he_normal', 
            padding='same',
            name=f'{name}_conv2'
        )(u)
        u = layers.Dropout(self.dropout_rate, name=f'{name}_dropout2')(u)
        
        return u
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass through the model."""
        return self.model(inputs, training=training)
    
    def get_config(self):
        """Return model configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_val,
            'num_classes': self.num_classes,
            'filters': self.filters,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()


def create_unet(input_shape=(128, 128, 3), num_classes=1, filters=16, dropout_rate=0.1):
    """
    Factory function to create a U-Net model.
    
    Args:
        input_shape (tuple): Input image shape
        num_classes (int): Number of output classes
        filters (int): Number of filters in first layer
        dropout_rate (float): Dropout rate
        
    Returns:
        Compiled U-Net model
    """
    model = UNet(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=filters,
        dropout_rate=dropout_rate
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Create and test the model
    model = create_unet(input_shape=(128, 128, 3))
    model.summary()
    
    # Test with dummy data
    dummy_input = tf.random.normal((1, 128, 128, 3))
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
