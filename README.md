# ML_Assignment
# Latent Feature Extraction for Musical Genres from Raw Audio

## Machine Learning Mini Project

### Team Members

| Narina Jeevan Naga Deep | PES1UG24CS293 |
| Nayan Niranjana Kaushik | PES1UG24CS294 |

---

## Project Information

**Project Title:** Latent Feature Extraction for Musical Genres from Raw Audio  
**Domain:** Machine Learning / Audio Classification  
**Reference Paper:** Latent Feature Extraction for Musical Genres from Raw Audio  
**Dataset:** GTZAN Music Genre Dataset  
**Model:** Deep Softmax Autoencoder (DSAE)  
**Programming Language:** Python  
**Libraries:** PyTorch, NumPy, Librosa, Scikit-learn

---

## 1. Project Overview

This project implements a **Deep Softmax Autoencoder (DSAE)** based approach for musical genre classification directly from raw audio signals.

Instead of relying on manually engineered audio features, the system learns a compact latent representation of the audio using an autoencoder. The learned representation is simultaneously used for genre classification.

The project focuses on four musical genres:

- Classical
- Jazz
- Metal
- Pop

The implementation follows the methodology of the reference paper and additionally applies classification-only fine-tuning to improve the final classification performance.

---

## 2. Problem Statement

Traditional music genre classification systems often depend on manually engineered audio features such as spectral, rhythmic, and timbral features.

The objective of this project is to investigate whether useful features for musical genre classification can be learned directly from raw audio using a **Deep Softmax Autoencoder**.

### Objectives

1. Process raw audio signals into fixed-length numerical representations.
2. Learn compact latent features using an autoencoder.
3. Use the learned latent representation for genre classification.
4. Compare the DSAE against a neural network baseline.
5. Improve classification performance using classification-only fine-tuning.
6. Evaluate the final model on a held-out test set.

---

## 3. Dataset

The project uses the **GTZAN Music Genre Dataset**.

The original GTZAN dataset contains 10 musical genres. Four genres were selected for this project:

- Classical
- Jazz
- Metal
- Pop

Each audio track is approximately 30 seconds long.

One invalid Jazz audio file was removed during dataset validation. The resulting dataset contains **399 valid audio tracks and 7980 processed examples**.

### Dataset Distribution

| Genre | Examples |
|---|---:|
| Classical | 2000 |
| Jazz | 1980 |
| Metal | 2000 |
| Pop | 2000 |
| **Total** | **7980** |

---

## 4. Data Preprocessing

The audio files are processed using **Librosa**.

### Preprocessing Pipeline

```text
Raw Audio
    ↓
Load at 22.05 kHz
    ↓
Take first 600000 samples
    ↓
Reshape into groups of 40 samples
    ↓
Average Pooling
    ↓
500-dimensional representation
    ↓
Create 20 segments per audio file
```

Each final example contains **500 features**.

Average pooling reduces the dimensionality of the raw audio signal while also providing regularization.

---

## 5. Dataset Split

The processed dataset is divided into training, development, and test sets.

| Dataset | Number of Samples |
|---|---:|
| Training | 5985 |
| Development | 997 |
| Test | 998 |
| **Total** | **7980** |

The test set is kept separate and is used only for final evaluation.

---

## 6. Baseline Model

A simple neural network is used as the baseline.

### Architecture

```text
Input: 500
   ↓
Hidden Layer: 128
   ↓
Tanh
   ↓
Output: 4 Classes
```

### Baseline Test Accuracy

**50.90%**

This provides a reference point for measuring the improvement obtained using the DSAE.

---

## 7. Deep Softmax Autoencoder

The main model is a **Deep Softmax Autoencoder (DSAE)** consisting of an encoder, decoder, and classifier.

### Encoder

**Architecture:**

```text
500 → 256 → 192 → 128 → 64
```

**Activations:**

- Sigmoid
- Tanh
- Dropout(0.2)
- ReLU
- ReLU

The encoder produces a **64-dimensional latent representation**.

### Decoder

**Architecture:**

```text
64 → 128 → 192 → 256 → 500
```

**Activations:**

- Sigmoid
- Sigmoid
- Dropout(0.2)
- ReLU
- ReLU

### Classifier

**Architecture:**

```text
64 → 32 → 16 → 4
```

The classifier uses Tanh activations in the hidden layers and a final linear layer for the four genre classes.

---

## 8. Training Objective

The DSAE is trained using both reconstruction and classification objectives.

### Total Loss

```text
Total Loss =
Reconstruction Loss
+
0.1 × Classification Loss
```

The reconstruction objective encourages the latent representation to preserve useful information from the input audio.

The classification objective encourages the latent representation to become useful for distinguishing musical genres.

---

## 9. DSAE Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 512 |
| Epochs | 200 |
| Dropout | 0.2 |
| Latent Dimension | 64 |
| Number of Classes | 4 |

The best DSAE model is saved as:

```text
dsae_best.pth
```

### DSAE Test Accuracy

**60.22%**

---

## 10. Classification Fine-Tuning

After training the DSAE, a classification-only fine-tuning stage is performed.

The pretrained DSAE checkpoint is loaded and further trained using only the classification loss.

### Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.00001 |
| Batch Size | 512 |
| Epochs | 50 |
| Loss | Cross Entropy |
| Model Selection | Best Development Accuracy |
| Random Seed | 42 |

The best model is selected using development-set accuracy.

The final fine-tuned model is saved as:

```text
dsae_finetuned.pth
```

---

## 11. Final Results

| Model | Test Accuracy |
|---|---:|
| Baseline Neural Network | **50.90%** |
| DSAE | **60.22%** |
| DSAE + Fine-Tuning | **62.83%** |
| Reference Paper DSAE | **65.30%** |

### Improvement

The final fine-tuned DSAE improves over the baseline by:

**62.83% - 50.90% = 11.93 percentage points**

It improves over the original DSAE by:

**62.83% - 60.22% = 2.61 percentage points**

The final implementation is **2.47 percentage points** below the 65.30% test accuracy reported in the reference paper.

---

## 12. Final Classification Performance

The final fine-tuned DSAE achieved an overall test accuracy of:

**62.83%**

### Classification Report

| Genre | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Classical | 0.7481 | 0.7840 | 0.7656 |
| Jazz | 0.6041 | 0.5968 | 0.6004 |
| Metal | 0.5149 | 0.4840 | 0.4990 |
| Pop | 0.6328 | 0.6480 | 0.6403 |
| **Overall Accuracy** | | | **0.6283** |

---

## 13. Confusion Matrix

The final confusion matrix is:

```text
[[196  33  17   4]
 [ 57 148  32  11]
 [  5  45 121  79]
 [  4  19  65 162]]
```

The model performs best on **Classical** music.

The largest confusion occurs between **Metal and Pop**, indicating that these genres have greater overlap in the learned representation.

---

## 14. Project Pipeline

```text
GTZAN Audio Dataset
        ↓
Genre Selection
        ↓
Audio Preprocessing
        ↓
Average Pooling
        ↓
500-Dimensional Input
        ↓
Train / Development / Test Split
        ↓
        ┌──────────────────┐
        │                  │
        ↓                  ↓
 Baseline Model        DSAE Model
        │                  │
        ↓                  ↓
 Baseline Result      Latent Features
                           │
                           ↓
                     Classification
                           │
                           ↓
                      DSAE Result
                           │
                           ↓
                Classification Fine-Tuning
                           │
                           ↓
                    Final Prediction
                           │
                           ↓
                       62.83%
```

---

## 15. Project Structure

```text
ML_Assignment/
│
├── Data/
│   ├── genres_original/
│   ├── images_original/
│   └── processed/
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_dev.npy
│       ├── y_dev.npy
│       ├── X_test.npy
│       └── y_test.npy
│
├── autoencoder.py
├── baseline_model.py
├── deep_softmax_autoencoder.py
├── dsae_best.pth
├── dsae_finetune.py
├── dsae_finetuned.pth
├── preprocess.py
└── README.md
```

---

## 16. File Description

| File | Purpose |
|---|---|
| `preprocess.py` | Processes raw audio and creates numerical input data |
| `baseline_model.py` | Implements the baseline neural network |
| `autoencoder.py` | Implements the vanilla autoencoder |
| `deep_softmax_autoencoder.py` | Implements and trains the DSAE |
| `dsae_best.pth` | Best trained DSAE checkpoint |
| `dsae_finetune.py` | Performs classification-only fine-tuning |
| `dsae_finetuned.pth` | Final fine-tuned DSAE checkpoint |
| `Data/processed/` | Contains processed train/dev/test arrays |

---

## 17. Installation

### Requirements

- Python 3.x
- NumPy
- PyTorch
- Librosa
- Scikit-learn

Install the required libraries:

```bash
pip install numpy torch librosa scikit-learn
```

---

## 18. How to Run

### Step 1: Prepare the Dataset

Place the GTZAN dataset inside:

```text
Data/genres_original/
```

The required genre folders are:

```text
classical/
jazz/
metal/
pop/
```

### Step 2: Preprocess the Audio

If the processed `.npy` files are not already available, run:

```bash
python preprocess.py
```

This generates the processed train, development, and test arrays.

### Step 3: Train the Baseline

Run:

```bash
python baseline_model.py
```

This trains and evaluates the baseline neural network.

### Step 4: Train the DSAE

Run:

```bash
python deep_softmax_autoencoder.py
```

This trains the Deep Softmax Autoencoder and saves:

```text
dsae_best.pth
```

### Step 5: Fine-Tune the DSAE

Run:

```bash
python dsae_finetune.py
```

This performs classification-only fine-tuning and saves:

```text
dsae_finetuned.pth
```

The final model achieves:

**Test Accuracy: 62.83%**

---

## 19. Reproducibility

The implementation uses:

```text
Random Seed = 42
```

The training, development, and test datasets are kept separate.

The development set is used for model selection, while the test set is used for the final evaluation.

The final reported test accuracy is:

**62.83%**

---

## 20. Conclusion

This project demonstrates the use of a **Deep Softmax Autoencoder** for learning latent features directly from raw audio for musical genre classification.

The baseline neural network achieved a test accuracy of **50.90%**.

The DSAE improved the performance to **60.22%**, showing the benefit of learning a compact latent representation.

Classification-only fine-tuning further improved the test accuracy to **62.83%**.

Therefore, the final system provides a substantial improvement over the baseline while approaching the **65.30%** test accuracy reported in the reference paper.

The results also show that Classical music is classified most effectively, while Metal and Pop remain more challenging due to their higher degree of confusion.

---

## 21. Reference

Sawhney, A., Vasavada, V., & Wang, W.

**"Latent Feature Extraction for Musical Genres from Raw Audio."**

NIPS 2018.
