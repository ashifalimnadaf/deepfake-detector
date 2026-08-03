"""
Inference Command-Line Script for Deepfake Detection.

This script allows users to run inference on individual images:
1. Loads the saved deepfake detection model.
2. Reads and preprocesses the target image (including face extraction if possible).
3. Evaluates the model prediction.
4. Outputs the final prediction: REAL or FAKE, along with the confidence score.
"""

import os
import argparse
# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf

# Import modular preprocessing
from preprocessing import preprocess_image

def predict_single_image(image_path, model_path):
    """
    Performs inference on a single image file and prints the classification result.
    
    Args:
        image_path (str): Path to the target image file.
        model_path (str): Path to the saved Keras model file.
        
    Returns:
        tuple: (prediction label as string, confidence score as float)
    """
    # 1. Verify file paths
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image file not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please train the model first.")
        
    # 2. Load the trained model
    print(f"Loading deepfake detection model from: {model_path}...")
    # Load model using standard Keras API, falling back to rebuilding architecture & loading weights if H5 config is missing
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Note: Loading full model configuration failed ({e}). Rebuilding architecture and loading weights...")
        from model import build_deepfake_detector_model
        model = build_deepfake_detector_model()
        
        # Keras 3 requires weights filenames to end in '.weights.h5' to use the modern weights loader.
        # If loading a legacy '.h5' file, copy it temporarily to use the correct parser.
        if model_path.endswith('.h5') and not model_path.endswith('.weights.h5'):
            import shutil
            temp_weights_path = model_path + ".weights.h5"
            shutil.copy2(model_path, temp_weights_path)
            try:
                model.load_weights(temp_weights_path)
            finally:
                if os.path.exists(temp_weights_path):
                    os.remove(temp_weights_path)
        else:
            model.load_weights(model_path)
    
    # 3. Preprocess the image
    print(f"Preprocessing target image: {image_path}...")
    preprocessed_img = preprocess_image(image_path, target_size=(300, 300), is_training=False)
    
    # Add batch dimension to match expected model input shape: (1, 300, 300, 3)
    img_input = np.expand_dims(preprocessed_img, axis=0)
    
    # 4. Perform model prediction
    print("Running inference...")
    prediction_prob = model.predict(img_input, verbose=0)[0][0]
    
    # 5. Interpret outputs
    # Label mapping: Real = 0 (low probability), Fake = 1 (high probability)
    if prediction_prob >= 0.5:
        prediction_label = "FAKE"
        confidence_percentage = prediction_prob * 100.0
    else:
        prediction_label = "REAL"
        confidence_percentage = (1.0 - prediction_prob) * 100.0
        
    print("\n" + "="*40)
    print(f"Prediction Result for: {os.path.basename(image_path)}")
    print(f"Classification:      {prediction_label}")
    print(f"Confidence Score:    {confidence_percentage:.2f}%")
    print(f"Raw Output Prob:     {prediction_prob:.6f}")
    print("="*40 + "\n")
    
    return prediction_label, confidence_percentage

def main():
    parser = argparse.ArgumentParser(description="Predict whether a single image is REAL or FAKE.")
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to analyze.')
    parser.add_argument('--model', type=str, default=os.path.join('saved_model', 'deepfake_detector_best.keras'),
                        help='Path to the trained model (.keras or .h5).')
    args = parser.parse_args()
    
    try:
        predict_single_image(args.image, args.model)
    except Exception as e:
        print(f"Error executing prediction: {e}")

if __name__ == "__main__":
    main()
