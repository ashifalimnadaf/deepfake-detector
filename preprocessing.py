"""
Preprocessing Module for Deepfake Detection.

This module contains functions to:
1. Load images from file paths.
2. Detect and extract faces using MediaPipe (high accuracy) with Haar Cascade fallback.
3. Apply data augmentations (flip, rotation, brightness, blur, compression, noise) for training.
4. Resize images to 300x300 and normalize pixel values to [0, 1] float32.
"""

import os
import random
import cv2
import numpy as np

# Initialize MediaPipe face detector safely
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        mp_face_detection = mp.solutions.face_detection
        # Use model_selection=1 for full-range images (typically faces further than 2m), 0 for close-up (within 2m)
        mp_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        print("MediaPipe Face Detection successfully initialized.")
    else:
        mp_detector = None
        print("Notice: MediaPipe solutions module not available in installed version. Face detection will fall back to OpenCV Haar Cascade.")
except Exception as e:
    mp_detector = None
    print(f"Notice: MediaPipe face detection could not be loaded ({e}). Face detection will fall back to OpenCV Haar Cascade.")

# Initialize Haar Cascade face detector using OpenCV's built-in model
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    print("Warning: OpenCV Haar Cascade XML could not be loaded. Face detection fallback might fail, falling back to full image.")

def extract_face(image, padding_ratio=0.15):
    """
    Detects and extracts the largest face in an image using MediaPipe Face Detection
    for superior accuracy and rotation invariance, falling back to OpenCV Haar Cascade.
    
    Args:
        image (numpy.ndarray): Input RGB image.
        padding_ratio (float): Padding ratio to add around the detected face box.
        
    Returns:
        numpy.ndarray: Cropped face image, or the original image if no face is detected.
    """
    height, width, _ = image.shape
    
    # 1. Try MediaPipe first
    if mp_detector is not None:
        try:
            # MediaPipe expects RGB image (which input image is)
            results = mp_detector.process(image)
            if results.detections:
                largest_det = None
                max_area = 0
                
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    w_px = int(bbox.width * width)
                    h_px = int(bbox.height * height)
                    area = w_px * h_px
                    if area > max_area:
                        max_area = area
                        largest_det = bbox
                        
                if largest_det is not None:
                    # Guard against negative values and handle coordinates
                    xmin = max(0, int(largest_det.xmin * width))
                    ymin = max(0, int(largest_det.ymin * height))
                    w_px = int(largest_det.width * width)
                    h_px = int(largest_det.height * height)
                    
                    # Add padding
                    pad_w = int(w_px * padding_ratio)
                    pad_h = int(h_px * padding_ratio)
                    
                    x1 = max(0, xmin - pad_w)
                    y1 = max(0, ymin - pad_h)
                    x2 = min(width, xmin + w_px + pad_w)
                    y2 = min(height, ymin + h_px + pad_h)
                    
                    cropped_face = image[y1:y2, x1:x2]
                    if cropped_face.size > 0:
                        return cropped_face
        except Exception as e:
            print(f"Warning: MediaPipe face extraction failed with error: {e}. Falling back to Haar Cascade.")

    # 2. Fallback to OpenCV Haar Cascade
    try:
        # Convert RGB image to grayscale for Haar Cascade detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            # Select the largest face by area
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Calculate padding around the face
            pad_w = int(w * padding_ratio)
            pad_h = int(h * padding_ratio)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(width, x + w + pad_w)
            y2 = min(height, y + h + pad_h)
            
            cropped_face = image[y1:y2, x1:x2]
            if cropped_face.size > 0:
                return cropped_face
    except Exception as e:
        print(f"Warning: Haar Cascade face extraction failed with error: {e}")
        
    return image

def augment_image(image):
    """
    Applies random augmentations to an image using OpenCV and numpy operations.
    
    Augmentations:
    - Random horizontal flip (50% chance).
    - Random rotation between -15 and 15 degrees (50% chance).
    - Random brightness adjustment (+/- 15%) (50% chance).
    - Random Gaussian Blur (30% chance) - simulates low-res deepfakes or face blending.
    - Random JPEG Compression artifacts (30% chance) - simulates internet sharing.
    - Random Gaussian Noise (30% chance) - regularizes the network.
    
    Args:
        image (numpy.ndarray): Input RGB image.
        
    Returns:
        numpy.ndarray: Augmented RGB image.
    """
    # 1. Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        
    # 2. Random rotation (-15 to 15 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-15.0, 15.0)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
    # 3. Random brightness adjustment (+/- 15%)
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.85, 1.15)
        # Multiply float image to prevent overflow, then clip and convert back
        temp = image.astype(np.float32) * brightness_factor
        image = np.clip(temp, 0.0, 255.0).astype(np.uint8)
        
    # 4. Random Gaussian Blur (30% chance)
    if random.random() > 0.7:
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
    # 5. Random JPEG Compression (30% chance)
    if random.random() > 0.7:
        quality = random.randint(50, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if result:
            image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            
    # 6. Random Gaussian Noise (30% chance)
    if random.random() > 0.7:
        row, col, ch = image.shape
        mean = 0
        var = random.uniform(10, 50)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image.astype(np.float32) + gauss
        image = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
        
    return image

def preprocess_image(file_path, target_size=(300, 300), is_training=False):
    """
    Loads, processes, and normalizes an image for deep learning model ingestion.
    
    Args:
        file_path (str): File path of the image.
        target_size (tuple): Target resizing dimensions (height, width).
        is_training (bool): If True, applies data augmentation.
        
    Returns:
        numpy.ndarray: Preprocessed float32 image normalized to [0, 1] with shape (300, 300, 3).
    """
    # 1. Load image in BGR
    img = cv2.imread(file_path)
    if img is None:
        # If image cannot be read, return a blank image of target size to avoid breaking training
        print(f"Warning: Failed to load image at {file_path}. Using a dummy blank image.")
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Extract face (optional but highly preferred for deepfake detection)
    try:
        img = extract_face(img)
    except Exception as e:
        # Fallback if face extraction throws an unexpected error
        print(f"Warning: Face extraction failed for {file_path} with error {e}. Using full image.")
        
    # 3. Apply training data augmentation
    if is_training:
        img = augment_image(img)
        
    # 4. Resize to target dimensions
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 5. Normalize pixel values to [0, 1] float32
    img_normalized = img.astype(np.float32) / 255.0
    
    return img_normalized

# Main verification block
if __name__ == "__main__":
    # Test preprocessing on a random matrix if run directly
    dummy_img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    cv2.imwrite("temp_test.jpg", dummy_img)
    try:
        proc_img = preprocess_image("temp_test.jpg", target_size=(300, 300), is_training=True)
        print("Preprocessing test image shape:", proc_img.shape)
        print("Preprocessing test image min/max value:", proc_img.min(), proc_img.max())
        print("Preprocessing verification complete.")
    finally:
        if os.path.exists("temp_test.jpg"):
            os.remove("temp_test.jpg")

