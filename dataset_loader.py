"""
Dataset Loader Module for Deepfake Detection.

This module is responsible for:
1. Scanning the dataset directory for 'real' and 'fake' subfolders.
2. Generating a small synthetic dataset if no data is present, allowing out-of-the-box pipeline validation.
3. Performing stratified splitting into training, validation, and testing sets.
4. Building optimized tf.data.Dataset pipelines for training, validation, and testing.
"""

import os
import glob
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

def generate_dummy_images(dataset_dir, num_real=50, num_fake=50, img_size=(256, 256)):
    """
    Generates synthetic dummy images for testing the pipeline when no dataset is present.
    
    To make them distinguishable, 'real' images will have green circles,
    and 'fake' images will have red squares, overlaid on random backgrounds.
    
    Args:
        dataset_dir (str): Root directory for the dataset.
        num_real (int): Number of real dummy images to generate.
        num_fake (int): Number of fake dummy images to generate.
        img_size (tuple): Height and width of the dummy images.
    """
    print(f"Dataset directory '{dataset_dir}' empty or not found. Generating dummy dataset for testing...")
    
    real_dir = os.path.join(dataset_dir, 'real')
    fake_dir = os.path.join(dataset_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Set seed for reproducible dummy data
    np.random.seed(42)
    
    # Generate REAL dummy images (Green circles on random backgrounds)
    for i in range(num_real):
        img = np.random.randint(50, 200, size=(img_size[0], img_size[1], 3), dtype=np.uint8)
        # Draw a green circle
        cv2.circle(img, (img_size[1] // 2, img_size[0] // 2), 60, (0, 220, 0), -1)
        # Add some text to simulate facial structures slightly
        cv2.putText(img, "REAL FACE", (30, img_size[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(real_dir, f"real_{i:03d}.jpg"), img)
        
    # Generate FAKE dummy images (Red squares on random backgrounds)
    for i in range(num_fake):
        img = np.random.randint(50, 200, size=(img_size[0], img_size[1], 3), dtype=np.uint8)
        # Draw a red square
        cv2.rectangle(img, (img_size[1] // 2 - 50, img_size[0] // 2 - 50), 
                      (img_size[1] // 2 + 50, img_size[0] // 2 + 50), (0, 0, 220), -1)
        # Add some text to simulate deepfake artifacts
        cv2.putText(img, "FAKE FACE", (30, img_size[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(fake_dir, f"fake_{i:03d}.jpg"), img)
        
    print(f"Dummy dataset generated: {num_real} real and {num_fake} fake images created.")

def load_dataset_paths(dataset_dir):
    """
    Scans the dataset directory and retrieves paths to all real and fake images.
    
    Args:
        dataset_dir (str): Root directory of the dataset containing 'real' and 'fake' subfolders.
        
    Returns:
        tuple: (list of image paths, list of integer labels)
               where 0 represents REAL and 1 represents FAKE.
    """
    real_dir = os.path.join(dataset_dir, 'real')
    fake_dir = os.path.join(dataset_dir, 'fake')
    
    # Supported image extensions
    valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    
    real_paths = []
    fake_paths = []
    
    if os.path.exists(real_dir) and os.path.exists(fake_dir):
        for ext in valid_extensions:
            real_paths.extend(glob.glob(os.path.join(real_dir, ext)))
            fake_paths.extend(glob.glob(os.path.join(fake_dir, ext)))
            
    # If no images are found, generate dummy images
    if len(real_paths) == 0 or len(fake_paths) == 0:
        generate_dummy_images(dataset_dir)
        real_paths = []
        fake_paths = []
        for ext in valid_extensions:
            real_paths.extend(glob.glob(os.path.join(real_dir, ext)))
            fake_paths.extend(glob.glob(os.path.join(fake_dir, ext)))
            
    paths = real_paths + fake_paths
    # Label mapping: REAL = 0, FAKE = 1
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    
    print(f"Dataset overview: Total images found: {len(paths)} (Real: {len(real_paths)}, Fake: {len(fake_paths)})")
    return paths, labels

def split_dataset(paths, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Splits dataset paths and labels into stratified train, validation, and test sets.
    
    Args:
        paths (list): List of image file paths.
        labels (list): List of class labels (0 or 1).
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of testing data.
        random_seed (int): Random state for reproducibility.
        
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Split ratios must sum up to 1.0"
    
    # First split off train set
    remaining_ratio = val_ratio + test_ratio
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        paths, labels, 
        test_size=remaining_ratio, 
        random_state=random_seed, 
        stratify=labels
    )
    
    # Split the remaining validation/test sets
    test_relative_ratio = test_ratio / remaining_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels, 
        test_size=test_relative_ratio, 
        random_state=random_seed, 
        stratify=val_test_labels
    )
    
    print(f"Splits generated successfully:")
    print(f"  Training set: {len(train_paths)} images")
    print(f"  Validation set: {len(val_paths)} images")
    print(f"  Testing set: {len(test_paths)} images")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def build_tf_dataset(paths, labels, preprocess_func, batch_size=16, is_training=False, buffer_size=tf.data.AUTOTUNE):
    """
    Builds an optimized tf.data.Dataset pipeline.
    
    Args:
        paths (list): List of image file paths.
        labels (list): List of class labels.
        preprocess_func (callable): Function imported from preprocessing module to preprocess images.
        batch_size (int): Size of training batches.
        is_training (bool): If True, applies data shuffling and augmentations.
        buffer_size (int): Prefetching buffer size.
        
    Returns:
        tf.data.Dataset: Processed and batched dataset.
    """
    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    # Parse and preprocess image paths on the fly
    def _parse_and_preprocess(path, label):
        # We wrap the preprocessing function using tf.py_function because face detection
        # and OpenCV operations run in standard Python and cannot be fully compiled to graph mode
        def _py_preprocess(p, l):
            # Decode bytes if path is passed as tensor string
            p_str = p.numpy().decode('utf-8')
            img_tensor = preprocess_func(p_str, is_training=is_training)
            return img_tensor, tf.cast(l, tf.float32)
        
        # Specify output shapes and types
        img_res, label_res = tf.py_function(
            _py_preprocess, 
            inp=[path, label], 
            Tout=[tf.float32, tf.float32]
        )
        # Re-establish tensor shapes for the graph
        img_res.set_shape([300, 300, 3])
        label_res.set_shape([])
        return img_res, label_res
    
    dataset = dataset.map(_parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        # Shuffle training dataset
        dataset = dataset.shuffle(buffer_size=len(paths))
        
    # Batch and prefetch for hardware acceleration
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    
    return dataset

# Main verification block for debugging purposes
if __name__ == "__main__":
    # Test dataset loader execution in isolation
    dummy_path = os.path.join(os.path.dirname(__file__), "dataset")
    p, l = load_dataset_paths(dummy_path)
    tr_p, tr_l, val_p, val_l, te_p, te_l = split_dataset(p, l)
    print("Dataset loader verification complete.")
