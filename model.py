"""
Model Module for Deepfake Detection.

This module is responsible for defining, assembling, and compiling the
Deepfake Detection CNN model using transfer learning with EfficientNetB0.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_deepfake_detector_model(input_shape=(300, 300, 3), learning_rate=1e-4):
    """
    Builds and compiles a transfer learning model based on EfficientNetB3.
    
    Args:
        input_shape (tuple): BGR/RGB input tensor shape (height, width, channels).
        learning_rate (float): Initial optimizer learning rate.
        
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    # Define the input layer matching the preprocessing output shape (normalized [0, 1])
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Keras EfficientNet expects pixel values in the range [0, 255] float.
    # Therefore, we rescale our normalized [0, 1] inputs back to [0, 255] range.
    rescaled_inputs = layers.Rescaling(scale=255.0, name="rescale_to_255")(inputs)
    
    # Instantiate the base EfficientNetB3 model pre-trained on ImageNet
    # We do not pass input_tensor to avoid tying the base model's inner graph to rescaled_inputs,
    # which can cause serialization issues when saving to legacy HDF5 formats.
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the convolutional base weights to preserve features learned from ImageNet
    base_model.trainable = False
    
    # Call the base model as a layer
    x = base_model(rescaled_inputs, training=False)
    
    # Add a global average pooling layer to collapse spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add a Dense hidden layer for feature representation
    x = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Apply Dropout for regularization to reduce overfitting
    x = layers.Dropout(0.4)(x)
    
    # Output layer for binary classification: 0 (REAL) or 1 (FAKE)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the complete Keras model
    model = models.Model(inputs=inputs, outputs=outputs, name="Deepfake_EfficientNetB3")
    
    # Compile the model with binary classification settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def unfreeze_model_for_finetuning(model, fine_tune_from_layer=-30, learning_rate=1e-5):
    """
    Unfreezes the top layers of the base model for fine-tuning.
    
    Args:
        model (tf.keras.Model): The built deepfake detector model.
        fine_tune_from_layer (int): Number of top layers to unfreeze in the base model 
                                    (e.g., -30 unfreezes the last 30 layers).
        learning_rate (float): A lower learning rate for fine-tuning to prevent destroying 
                               pre-trained weights.
                               
    Returns:
        tf.keras.Model: Re-compiled model ready for fine-tuning.
    """
    # Locate the EfficientNet base model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or layer.name.startswith("efficientnet"):
            base_model = layer
            break
            
    if base_model is None:
        print("Warning: EfficientNet base layer not found. Fine-tuning setup aborted.")
        return model
        
    # Unfreeze the base model entirely first
    base_model.trainable = True
    
    # Freeze all layers before the fine_tune_from_layer index
    # We use negative indexing relative to base_model.layers
    num_layers = len(base_model.layers)
    cutoff = num_layers + fine_tune_from_layer if fine_tune_from_layer < 0 else fine_tune_from_layer
    
    for i, layer in enumerate(base_model.layers[:cutoff]):
        layer.trainable = False
        
    print(f"Base model layers unfrozen starting from index {cutoff} ({base_model.layers[cutoff].name}).")
    print(f"Total trainable layers in base: {sum([1 for l in base_model.layers if l.trainable])}")
    
    # Recompile model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# Main verification block
if __name__ == "__main__":
    # Build model and print architecture summary
    model = build_deepfake_detector_model()
    model.summary()
    print("Initial model trainable weights count:", len(model.trainable_weights))
    
    # Try unfreezing some layers
    model = unfreeze_model_for_finetuning(model, fine_tune_from_layer=-30)
    print("Fine-tuning model trainable weights count:", len(model.trainable_weights))
