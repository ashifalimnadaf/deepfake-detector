# DeepShield AI - Deepfake Image Detection System

This repository contains a production-grade, modular Deepfake Image Detection system using Python, TensorFlow/Keras, and Streamlit. The model is built on **EfficientNetB0** utilizing transfer learning to extract robust facial representations and classify images as **REAL** or **FAKE**.

---

## 📁 Project Structure

```
Deepfake_Detection/
│
├── dataset/                  # Dataset folder (subdivided into real/ and fake/)
├── models/                   # Directory to save temporary model checkpoints during training
├── saved_model/              # Location of final models and generated performance plots
│   ├── deepfake_detector_best.keras  # Modern Keras model file
│   ├── deepfake_detector_best.h5     # Legacy Keras HDF5 model file
│   ├── training_history_curves.png   # Accuracy and loss plot
│   ├── confusion_matrix.png          # Visual confusion matrix on the test set
│   ├── roc_curve.png                 # Receiver Operating Characteristic plot
│   └── evaluation_results.json       # JSON file containing accuracy, recall, precision, F1, and AUC
│
├── dataset_loader.py         # Directory scanner, stratified splitter, and tf.data pipelines
├── preprocessing.py          # OpenCV Haar Cascade face detection, resizing, and augmentations
├── model.py                  # Keras model assembler and fine-tuning configurations
├── train.py                  # Training pipeline, learning rate decay, and test evaluator
├── predict.py                # Command-line tool to run inference on a single image file
├── utils.py                  # Helper functions for plots and quantitative report generation
├── app.py                    # Streamlit dashboard GUI with interactive animations
├── requirements.txt          # Python packages list
└── README.md                 # Project user manual (this document)
```

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.10+** installed. Follow these steps to set up the environment:

1. **Navigate to the Project Directory:**
   ```bash
   cd Deepfake_Detection
   ```

2. **Create a Virtual Environment:**
   * **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **Linux/macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Upgrade Pip and Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Setup

The pipeline expects a folder named `dataset` structured as follows:
```
dataset/
├── real/
│   ├── image1.jpg
│   └── image2.png
└── fake/
    ├── image3.jpg
    └── image4.webp
```

> [!TIP]
> **Out-of-the-box Test Run (No Dataset Needed):** 
> If the `dataset` directory is empty or missing, running the pipeline will **automatically generate a small synthetic dataset** containing 50 real images (green circles) and 50 fake images (red squares) overlayed on random backgrounds. This allows you to verify that the training, checkpointing, and evaluation code works immediately before downloading large datasets.

---

## 🚀 Running the Pipeline

### 1. Train the Model
Run the main training script. It automatically detects and uses a CUDA-capable GPU if available (configuring memory growth), handles datasets splits (70% train, 15% val, 15% test), implements early stopping and learning rate reduction, and saves evaluation assets to `saved_model/`.

```bash
python train.py --epochs 20 --batch_size 16 --learning_rate 0.0001
```

**Key Training CLI Options:**
* `--dataset_dir`: Path to the image folder (default: `dataset`).
* `--saved_model_dir`: Path to save output assets (default: `saved_model`).
* `--epochs`: Maximum number of training epochs (default: `20`).
* `--batch_size`: Size of data batches (default: `16`).
* `--learning_rate`: Initial Adam learning rate (default: `0.0001`).

### 2. Predict via Command Line
Evaluate any individual image file using the prediction utility:

```bash
python predict.py --image path/to/image.jpg --model saved_model/deepfake_detector_best.keras
```

**Output format:**
```
========================================
Prediction Result for: sample.jpg
Classification:      FAKE
Confidence Score:    94.52%
Raw Output Prob:     0.945231
========================================
```

---

## 🖥️ Streamlit Web Dashboard

Start the high-fidelity graphical user interface using Streamlit:

```bash
streamlit run app.py
```

### Dashboard Features:
* **Interactive File Upload:** Drag & drop images directly.
* **Annotated Previews:** View the uploaded image with real-time face detection bounding boxes.
* **Glow Badges & Indicators:** Glowing green or magenta indicators based on classifications.
* **Hardware & Stats Panel:** View system parameters (GPU vs CPU), target image metrics, and pipeline details.
