"""
Utilities Module for Deepfake Detection.

This module provides tools to:
1. Plot and save training history curves (loss and accuracy).
2. Generate and plot the Confusion Matrix.
3. Generate and plot the ROC Curve (and calculate AUC).
4. Calculate and save comprehensive evaluation metrics (Precision, Recall, F1-score, AUC) to a JSON file.
"""

import os
import json
import matplotlib
# Use 'Agg' backend to allow headless saving of plots without display UI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    f1_score, 
    precision_score, 
    recall_score
)

def plot_training_curves(history, save_dir):
    """
    Plots training and validation accuracy and loss curves, and saves the plot as an image.
    
    Args:
        history (dict or keras.callbacks.History): History object returned by model.fit().
        save_dir (str): Directory where the plot image should be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Retrieve data from history dictionary
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
        
    epochs = range(1, len(hist_dict['loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_dict.get('accuracy', []), 'b-o', label='Training Acc')
    plt.plot(epochs, hist_dict.get('val_accuracy', []), 'r-s', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist_dict.get('loss', []), 'b-o', label='Training Loss')
    plt.plot(epochs, hist_dict.get('val_loss', []), 'r-s', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history_curves.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training history curves to: {plot_path}")

def plot_confusion_matrix(y_true, y_pred_prob, save_dir, threshold=0.5):
    """
    Calculates, prints, and plots the confusion matrix, then saves the plot.
    
    Args:
        y_true (numpy.ndarray): True binary labels (0 or 1).
        y_pred_prob (numpy.ndarray): Predicted probabilities (range [0, 1]).
        save_dir (str): Directory where the confusion matrix image should be saved.
        threshold (float): Classification decision boundary threshold.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Apply classification threshold
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Plotting Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['REAL', 'FAKE']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Annotate counts inside the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14)
                     
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved confusion matrix plot to: {plot_path}")
    print(f"Confusion Matrix values: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

def plot_roc_curve(y_true, y_pred_prob, save_dir):
    """
    Calculates, plots, and saves the ROC (Receiver Operating Characteristic) curve.
    
    Args:
        y_true (numpy.ndarray): True binary labels (0 or 1).
        y_pred_prob (numpy.ndarray): Predicted probabilities.
        save_dir (str): Directory where the ROC curve image should be saved.
        
    Returns:
        float: Calculated Area Under Curve (AUC) score.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved ROC curve plot to: {plot_path}")
    return roc_auc

def calculate_and_save_metrics(y_true, y_pred_prob, loss, accuracy, save_dir, threshold=0.5):
    """
    Calculates detailed evaluation metrics and saves them as a structured JSON file.
    
    Metrics: Loss, Accuracy, Precision, Recall, F1-score, and ROC AUC.
    
    Args:
        y_true (numpy.ndarray): True binary labels (0 or 1).
        y_pred_prob (numpy.ndarray): Predicted probabilities.
        loss (float): Loss value returned by the model evaluator.
        accuracy (float): Accuracy value returned by the model evaluator.
        save_dir (str): Directory where the metrics JSON file should be saved.
        threshold (float): Classification decision boundary threshold.
    """
    os.makedirs(save_dir, exist_ok=True)
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Calculate Sklearn metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Generate human-readable classification report text
    report_text = classification_report(y_true, y_pred, target_names=['REAL', 'FAKE'], zero_division=0)
    print("\nClassification Report:")
    print(report_text)
    
    # Package metrics into dictionary
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "threshold": float(threshold)
    }
    
    # Save to JSON
    json_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved evaluation metrics to: {json_path}")
    return metrics

# Main verification block
if __name__ == "__main__":
    # Test utilities with dummy random predictions
    y_t = np.array([0]*20 + [1]*20)
    y_p = np.random.uniform(0.1, 0.9, size=(40,))
    
    test_dir = "./saved_model_test"
    plot_training_curves({"loss": [0.6, 0.4, 0.3], "val_loss": [0.7, 0.5, 0.4], 
                          "accuracy": [0.6, 0.75, 0.82], "val_accuracy": [0.55, 0.70, 0.78]}, test_dir)
    plot_confusion_matrix(y_t, y_p, test_dir)
    plot_roc_curve(y_t, y_p, test_dir)
    calculate_and_save_metrics(y_t, y_p, 0.32, 0.81, test_dir)
    
    # Clean up test outputs
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print("Cleaned up test directories.")
