# -Handwritten-Digits-Recognizer-
# ✍️ Handwritten Digit Recognizer using Deep Learning (MNIST)

## 📘 Introduction

The **Handwritten Digit Recognizer** is an end-to-end AI application that classifies handwritten digits (0–9) using **Convolutional Neural Networks (CNNs)** trained on the MNIST dataset. It showcases how deep learning can be used to solve real-world computer vision problems through an interactive and user-friendly interface.

Built collaboratively, the project covers every stage of a machine learning pipeline—from **data preprocessing** to **model deployment**. It culminates in an intuitive **Gradio web interface** that allows users to draw digits and receive real-time predictions.

---

## 📚 Dataset Overview

- **Dataset Name:** MNIST (Modified National Institute of Standards and Technology)
- **Source:** `keras.datasets`
- **Training Images:** 60,000
- **Testing Images:** 10,000
- **Image Size:** 28x28 pixels, grayscale
- **Labels:** Digits from 0 to 9

Each sample is a centered grayscale image of a handwritten digit, ideal for classification using deep learning.

---

## 🧱 Project Modules 

###  Data Handling & Preprocessing

- Loaded the dataset via `keras.datasets`
- Normalized pixel values ([0–255] → [0–1])
- Reshaped data to 28x28x1 to include a color channel
- One-hot encoded labels using `to_categorical`
- Split data into training and validation sets

**Tools:** NumPy, TensorFlow/Keras

---

###  CNN Architecture Design

- Built a deep CNN with layers:
  - `Conv2D` for feature extraction (ReLU)
  - `MaxPooling2D` for downsampling
  - `Dropout` for regularization
  - `Flatten` + `Dense` for classification
- Output layer: `Softmax` for 10-class probability
- Compiled model with:
  - Optimizer: **Adam**
  - Loss: `categorical_crossentropy`
  - Metric: `accuracy`

**Initial Hyperparameters:**
- Batch Size: 64
- Epochs: 15

---

###  Training & Evaluation

- Trained CNN on preprocessed data
- Plotted:
  - Accuracy vs Epochs
  - Loss vs Epochs
- Evaluated model on test set:
  - Final accuracy
  - Confusion matrix

**Tools:** Matplotlib, Scikit-learn, Keras Callbacks

---

###  Error Analysis & Model Improvement

- Analyzed misclassified samples
- Identified difficult cases (e.g., ambiguous strokes)
- Enhanced model by:
  - Adding `BatchNormalization`
  - Adjusting dropout rates (0.25 → 0.3)
  - Increasing filters in `Conv2D` layers
  - Using a lower learning rate with more training epochs

**Outcome:** Improved generalization and performance on edge cases

---

###  Deployment with Gradio UI

- Created an interactive **Gradio** interface
- UI Features:
  - 28x28 canvas to draw digits
  - Predict button to show output with confidence
  - Clear button to reset canvas
- Integrated trained `.h5` model
- Enabled fast prediction and lightweight access

**Benefits of Gradio:**
- No frontend coding required
- Shareable public links for testing
- Simple and responsive design

---

## 📊 Results Summary

| Metric              | Value       |
|---------------------|-------------|
| Training Accuracy   | ~99%        |
| Validation Accuracy | ~98%        |
| Test Accuracy       | ~99.2%      |
| Model Loss          | Low & Stable|
| Misclassifications  | Significantly Reduced After Tuning |

The final CNN model showed excellent performance across all datasets and generalized well to unseen, user-drawn digits via the interface.

---

## 🎯 Project Goals Recap

✅ Build a robust CNN for digit recognition  
✅ Understand and implement the end-to-end deep learning workflow  
✅ Visualize training progress and evaluate model accuracy  
✅ Deploy an intuitive UI for real-time digit recognition  

---

## 🔧 Technologies & Tools

- **Languages:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn, Gradio  
- **Platforms:** Jupyter Notebook, Google Colab  
- **UI Framework:** Gradio

---

## 🛠 Sample Architecture

