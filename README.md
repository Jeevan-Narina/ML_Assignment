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

The implementation follows the methodology of the reference paper and additionally applies classification-only fine-tuning, MFCC-based classification, and DSAE latent-feature fusion experiments to improve and compare the final classification performance.

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
6. Extract MFCC features as an additional engineered feature representation.
7. Evaluate MFCC features using an RBF-SVM classifier.
8. Combine DSAE latent features with MFCC features and evaluate the fused representation.
9. Evaluate the final models on a held-out test set.

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

### Processed Raw Feature Shapes

| Dataset | Feature Shape |
|---|---:|
| Training | (5985, 500) |
| Development | (997, 500) |
| Test | (998, 500) |

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

The best development accuracy was **62.29% at epoch 47**.

The final fine-tuned model is saved as:

```text
dsae_finetuned.pth
```

The final fine-tuned DSAE achieved a test accuracy of:

**62.83%**

---

## 11. MFCC Feature Extraction

In addition to the raw-audio representation used by the DSAE, **Mel-Frequency Cepstral Coefficients (MFCCs)** are extracted as a conventional engineered audio feature representation.

The extracted MFCC representation contains **78 features per example**.

### MFCC Feature Shapes

| Dataset | Feature Shape |
|---|---:|
| Training | (5985, 78) |
| Development | (997, 78) |
| Test | (998, 78) |

The generated MFCC features are saved as:

```text
Data/processed/mfcc_train.npy
Data/processed/mfcc_dev.npy
Data/processed/mfcc_test.npy
```

---

## 12. MFCC + RBF-SVM

The extracted MFCC features are evaluated using an **RBF-kernel Support Vector Machine (SVM)**.

### Best Configuration

| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 1 |
| Gamma | scale |

### Classification Accuracy

| Dataset | Accuracy |
|---|---:|
| Training | 97.53% |
| Development | 95.49% |
| Test | **95.09%** |

### Classification Report

| Genre | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Classical | 0.9266 | 0.9600 | 0.9430 |
| Jazz | 0.9402 | 0.8871 | 0.9129 |
| Metal | 1.0000 | 0.9720 | 0.9858 |
| Pop | 0.9389 | 0.9840 | 0.9609 |

### Confusion Matrix

```text
[[240   9   0   1]
 [ 17 220   0  11]
 [  1   2 243   4]
 [  1   3   0 246]]
```

The MFCC + RBF-SVM experiment provides a strong conventional-feature baseline for comparison with the DSAE-based approach.

---

## 13. DSAE + MFCC Feature Fusion

To evaluate whether the learned DSAE representation contains complementary information to conventional audio features, the **64-dimensional DSAE latent representation** is concatenated with the **78-dimensional MFCC representation**.

### Fused Feature Representation

```text
DSAE Latent Features: 64
          +
MFCC Features: 78
          ↓
Combined Representation: 142
```

### Feature Shapes

| Dataset | Feature Shape |
|---|---:|
| Training | (5985, 142) |
| Development | (997, 142) |
| Test | (998, 142) |

The fused features are evaluated using an RBF-SVM classifier.

### Best Configuration

| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 100 |
| Gamma | 0.001 |

### Classification Accuracy

| Dataset | Accuracy |
|---|---:|
| Training | 98.81% |
| Development | 91.68% |
| Test | **90.48%** |

### Classification Report

| Genre | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Classical | 0.9170 | 0.8400 | 0.8768 |
| Jazz | 0.8083 | 0.8669 | 0.8366 |
| Metal | 0.9878 | 0.9680 | 0.9778 |
| Pop | 0.9147 | 0.9440 | 0.9291 |

### Confusion Matrix

```text
[[210  33   0   7]
 [ 16 215   2  15]
 [  2   6 242   0]
 [  1  12   1 236]]
```

The fusion experiment shows that combining learned DSAE latent features with MFCC features provides a strong representation, although its test performance is lower than the MFCC-only RBF-SVM result.

---

## 14. Final Results

| Model | Test Accuracy |
|---|---:|
| Baseline Neural Network | **50.90%** |
| DSAE | **60.22%** |
| DSAE + Fine-Tuning | **62.83%** |
| MFCC + RBF-SVM | **95.09%** |
| DSAE + MFCC + RBF-SVM | **90.48%** |
| Reference Paper DSAE | **65.30%** |

### Improvement

The final fine-tuned DSAE improves over the baseline by:

**62.83% - 50.90% = 11.93 percentage points**

It improves over the original DSAE by:

**62.83% - 60.22% = 2.61 percentage points**

The final implementation is **2.47 percentage points** below the 65.30% test accuracy reported in the reference paper.

The additional MFCC experiment achieved **95.09%** test accuracy using an RBF-SVM, while the DSAE + MFCC fused representation achieved **90.48%**.

The DSAE remains the main learned representation in the project, while the MFCC experiments provide additional feature-engineering and comparison baselines.

---

## 15. Final Classification Performance

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

## 16. Confusion Matrix

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

## 17. Project Pipeline

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
        ┌──────────────────────────────┐
        │                              │
        ↓                              ↓
 Baseline Model                    DSAE Model
        │                              │
        ↓                              ↓
 Baseline Result                64-D Latent Features
                                       │
                                       ├───────────────┐
                                       │               │
                                       ↓               ↓
                                 Classification     MFCC Features
                                       │               │
                                       ↓               ↓
                                  DSAE Result      Feature Fusion
                                       │               │
                                       ↓               ↓
                              Classification       RBF-SVM
                                Fine-Tuning            │
                                       │               ↓
                                       ↓          Fusion Result
                                  Final DSAE
                                  Prediction
                                       │
                                       ↓
                                    62.83%

MFCC Branch
    ↓
MFCC Feature Extraction
    ↓
RBF-SVM Classification
    ↓
95.09%
```

---

## 18. Project Structure

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
│       ├── y_test.npy
│       ├── mfcc_train.npy
│       ├── mfcc_dev.npy
│       └── mfcc_test.npy
│
├── autoencoder.py
├── baseline_model.py
├── deep_softmax_autoencoder.py
├── dsae_best.pth
├── dsae_finetune.py
├── dsae_finetuned.pth
├── dsae_mfcc_fusion.py
├── mfcc_features.py
├── mfcc_svm.py
├── preprocess.py
└── README.md
```

---

## 19. File Description

| File | Purpose |
|---|---|
| `preprocess.py` | Processes raw audio and creates numerical input data |
| `baseline_model.py` | Implements the baseline neural network |
| `autoencoder.py` | Implements the vanilla autoencoder |
| `deep_softmax_autoencoder.py` | Implements and trains the DSAE |
| `dsae_best.pth` | Best trained DSAE checkpoint |
| `dsae_finetune.py` | Performs classification-only fine-tuning |
| `dsae_finetuned.pth` | Final fine-tuned DSAE checkpoint |
| `mfcc_features.py` | Extracts and saves MFCC features |
| `mfcc_svm.py` | Trains and evaluates the RBF-SVM using MFCC features |
| `dsae_mfcc_fusion.py` | Combines DSAE latent features with MFCC features and evaluates the fused representation |
| `Data/processed/` | Contains processed train/dev/test and MFCC arrays |

---

## 20. Installation

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

## 21. How to Run

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

### Step 6: Extract MFCC Features

Run:

```bash
python mfcc_features.py
```

This generates:

```text
Data/processed/mfcc_train.npy
Data/processed/mfcc_dev.npy
Data/processed/mfcc_test.npy
```

### Step 7: Train the MFCC + RBF-SVM Model

Run:

```bash
python mfcc_svm.py
```

The best configuration uses:

```text
C = 1
Gamma = scale
```

The test accuracy is:

**95.09%**

### Step 8: Evaluate DSAE + MFCC Fusion

Run:

```bash
python dsae_mfcc_fusion.py
```

The best configuration uses:

```text
C = 100
Gamma = 0.001
```

The test accuracy is:

**90.48%**

---

## 22. Reproducibility

The implementation uses:

```text
Random Seed = 42
```

The same processed train, development, and test split is used across the experiments.

The development set is used for model selection and hyperparameter evaluation, while the test set is used for final evaluation.

The final reported test accuracies are:

```text
DSAE + Fine-Tuning       = 62.83%
MFCC + RBF-SVM           = 95.09%
DSAE + MFCC + RBF-SVM    = 90.48%
```

---

## 23. Conclusion

This project demonstrates the use of a **Deep Softmax Autoencoder** for learning latent features directly from raw audio for musical genre classification.

The baseline neural network achieved a test accuracy of **50.90%**.

The DSAE improved the performance to **60.22%**, showing the benefit of learning a compact latent representation.

Classification-only fine-tuning further improved the test accuracy to **62.83%**.

Therefore, the final system provides a substantial improvement over the baseline while approaching the **65.30%** test accuracy reported in the reference paper.

Additional experiments were performed using conventional **MFCC features**. The MFCC + RBF-SVM model achieved **95.09%** test accuracy.

The combination of the **64-dimensional DSAE latent representation** and **78-dimensional MFCC representation** achieved **90.48%** test accuracy using an RBF-SVM.

These experiments provide a comparison between learned raw-audio representations and conventional engineered audio features, while keeping the **DSAE as the main representation-learning approach** of the project.

---

## 24. Reference

Sawhney, A., Vasavada, V., & Wang, W.

**"Latent Feature Extraction for Musical Genres from Raw Audio."**

NIPS 2018.
