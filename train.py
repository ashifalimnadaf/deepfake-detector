"""
Training Pipeline for Deepfake Detection.

This script coordinates the training workflow:
1. Initializes random seeds for reproducibility.
2. Checks and configures GPU environments (CUDA).
3. Scans, splits, and loads the dataset.
4. Compiles the EfficientNetB0-based transfer learning model.
5. Employs Keras callbacks (Checkpointing, EarlyStopping, ReduceLROnPlateau).
6. Trains the model and evaluates it against the test set.
7. Saves the trained model to both '.keras' and '.h5' formats.
8. Generates and stores evaluation metrics (JSON, plots).
"""

import os
import argparse
import random
import numpy as np
import tensorflow as tf

# Import modular components
from dataset_loader import load_dataset_paths, split_dataset, build_tf_dataset
from preprocessing import preprocess_image
from model import build_deepfake_detector_model
from utils import plot_training_curves, plot_confusion_matrix, plot_roc_curve, calculate_and_save_metrics

def set_reproducibility_seed(seed=42):
    """
    Sets random seeds across Python, NumPy, and TensorFlow to ensure reproducible results.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Enable deterministic operations in TensorFlow if supported
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"Random seed set to {seed} for reproducibility.")

def configure_gpu_device():
    """
    Checks for available CUDA-capable GPUs and configures memory growth
    to prevent allocating the entire VRAM immediately.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"CUDA GPU detected: Using {len(gpus)} GPU(s) with memory growth enabled.")
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")
    else:
        print("No CUDA GPU detected. Running training on CPU.")

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train an EfficientNetB0 Deepfake Image Detector.")
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Path to dataset directory containing real/ and fake/ folders.')
    parser.add_argument('--saved_model_dir', type=str, default='saved_model',
                        help='Directory to save trained models and evaluation assets.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed value.')
    return parser.parse_args()

def main():
    # Parse CLI args
    args = parse_arguments()
    
    # 1. Set seed and setup GPU
    set_reproducibility_seed(args.seed)
    configure_gpu_device()
    
    # Ensure saved model directory exists
    os.makedirs(args.saved_model_dir, exist_ok=True)
    
    # 2. Scan and split dataset
    paths, labels = load_dataset_paths(args.dataset_dir)
    train_p, train_l, val_p, val_l, test_p, test_l = split_dataset(
        paths, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=args.seed
    )
    
    # 3. Build optimized tf.data.Dataset loaders
    print("\nPreparing TensorFlow data pipelines...")
    train_dataset = build_tf_dataset(
        train_p, train_l, preprocess_image, batch_size=args.batch_size, is_training=True
    )
    val_dataset = build_tf_dataset(
        val_p, val_l, preprocess_image, batch_size=args.batch_size, is_training=False
    )
    test_dataset = build_tf_dataset(
        test_p, test_l, preprocess_image, batch_size=args.batch_size, is_training=False
    )
    
    # 4. Initialize and compile transfer learning model
    print("\nAssembling EfficientNetB3 transfer learning model...")
    model = build_deepfake_detector_model(input_shape=(300, 300, 3), learning_rate=args.learning_rate)
    model.summary()
    
    # 5. Define training callbacks for Stage 1 (Warmup)
    checkpoint_filepath = os.path.join(args.saved_model_dir, "best_checkpoint.keras")
    
    callbacks_stage1 = [
        # Checkpoint callback: saves the model with best validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Early Stopping: halts training if validation loss doesn't improve for 5 consecutive epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
    ]
    
    # Divide epochs between Stage 1 and Stage 2
    stage1_epochs = max(1, args.epochs // 2)
    stage2_epochs = max(1, args.epochs - stage1_epochs)
    
    # 6. Execute Stage 1 Model Training (Classifier Warmup)
    print(f"\nStarting Stage 1 (Warmup) training for {stage1_epochs} epochs...")
    history_warmup = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1
    )
    
    # 6.2. Execute Stage 2 Model Training (Fine-Tuning)
    print("\nPreparing for Stage 2 (Fine-Tuning)...")
    # Load the best weights from Stage 1 warm-up to start fine-tuning
    if os.path.exists(checkpoint_filepath):
        print(f"Loading best Stage 1 weights from {checkpoint_filepath} for fine-tuning...")
        model = tf.keras.models.load_model(checkpoint_filepath)
        
    from model import unfreeze_model_for_finetuning
    # Unfreeze the top layers and set a smaller learning rate for fine-tuning
    model = unfreeze_model_for_finetuning(model, fine_tune_from_layer=-30, learning_rate=args.learning_rate * 0.1)
    
    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\nStarting Stage 2 (Fine-Tuning) training for {stage2_epochs} epochs...")
    history_finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=stage2_epochs,
        callbacks=callbacks_stage2
    )
    print("\nTraining completed successfully.")
    
    # Load the absolute best checkpoint model for final saving and evaluation
    if os.path.exists(checkpoint_filepath):
        print(f"Loading the absolute best model from {checkpoint_filepath} for final serialization and test set evaluation...")
        model = tf.keras.models.load_model(checkpoint_filepath)
        
    # 7. Save model in both Keras formats: .keras and .h5
    keras_save_path = os.path.join(args.saved_model_dir, "deepfake_detector_best.keras")
    h5_save_path = os.path.join(args.saved_model_dir, "deepfake_detector_best.h5")
    
    # Save the best model in native Keras format (retains compilation state)
    model.save(keras_save_path)
    print(f"Saved best model in .keras format to: {keras_save_path}")
    
    # Save the best model in legacy H5 format (disables compilation to avoid optimizer serialization bugs)
    if os.path.exists(checkpoint_filepath):
        print(f"Loading best weights with compile=False to save legacy H5 model...")
        h5_temp_model = tf.keras.models.load_model(checkpoint_filepath, compile=False)
        try:
            h5_temp_model.save(h5_save_path)
            print(f"Saved best model in .h5 format to: {h5_save_path}")
        except Exception as e:
            print(f"\n[Warning] Keras 3 could not serialize full EfficientNet model config to legacy H5 format: {e}")
            print("Saving model weights to legacy .h5 format instead. The architecture will be rebuilt on load.")
            # Keras 3 requires weights filenames to end in '.weights.h5'. We save from the active in-memory model (which preserves custom layer names) and then rename.
            weights_h5_path = h5_save_path.replace(".h5", ".weights.h5")
            model.save_weights(weights_h5_path)
            if os.path.exists(h5_save_path):
                os.remove(h5_save_path)
            os.rename(weights_h5_path, h5_save_path)
            print(f"Saved best model weights in .h5 format to: {h5_save_path}")
    else:
        # Fallback if checkpoint doesn't exist
        try:
            model.save(h5_save_path, include_optimizer=False)
            print(f"Saved model in .h5 format to: {h5_save_path}")
        except Exception as e:
            print(f"\n[Warning] Keras 3 could not serialize model config to legacy H5 format: {e}")
            weights_h5_path = h5_save_path.replace(".h5", ".weights.h5")
            model.save_weights(weights_h5_path)
            if os.path.exists(h5_save_path):
                os.remove(h5_save_path)
            os.rename(weights_h5_path, h5_save_path)
            print(f"Saved model weights in .h5 format to: {h5_save_path}")
    
    # 8. Run evaluation on testing split
    print("\nRunning model evaluation on the test set...")
    
    y_true_list = []
    y_pred_list = []
    
    # Iterate through testing batches to collect predictions and ground truth labels safely
    for batch_imgs, batch_labels in test_dataset:
        batch_preds = model.predict(batch_imgs, verbose=0)
        y_true_list.extend(batch_labels.numpy())
        y_pred_list.extend(batch_preds.flatten())
        
    y_true = np.array(y_true_list)
    y_pred_prob = np.array(y_pred_list)
    
    # Run evaluation loss and accuracy metrics
    evaluation_scores = model.evaluate(test_dataset, verbose=0)
    # evaluation_scores yields [loss, accuracy, precision, recall, auc] based on compiled metrics
    test_loss = evaluation_scores[0]
    test_acc = evaluation_scores[1]
    
    # Combine training histories for plotting
    combined_history = {}
    for key in history_warmup.history.keys():
        combined_history[key] = history_warmup.history[key] + history_finetune.history.get(key, [])
        
    # 9. Plot and save visual assets using utils.py
    print("\nGenerating and saving evaluation assets...")
    plot_training_curves(combined_history, args.saved_model_dir)
    plot_confusion_matrix(y_true, y_pred_prob, args.saved_model_dir)
    plot_roc_curve(y_true, y_pred_prob, args.saved_model_dir)
    calculate_and_save_metrics(y_true, y_pred_prob, test_loss, test_acc, args.saved_model_dir)
    
    print("\nAll evaluation reports and models have been successfully saved.")
    
if __name__ == "__main__":
    main()
