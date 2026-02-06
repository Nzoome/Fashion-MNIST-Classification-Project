# Fashion-MNIST Image Classification
## CNN vs Random Forest Comparison (Train + Test Evaluation)

## Overview
This project compares two approaches for classifying clothing images from the **Fashion-MNIST** dataset:

- **Convolutional Neural Network (CNN)** (deep learning baseline)
- **Random Forest (RF)** (classical machine learning baseline on flattened pixels)

Both models are evaluated on **training and test sets** using:
- Accuracy
- Confusion matrices
- Precision / Recall / F1 (classification reports)
- Training time
- Error analysis on visually similar classes (e.g., *Shirt vs T-shirt/top*)

---

## Dataset
Fashion-MNIST contains **70,000 grayscale images** (28×28) across **10 classes**:

- Training set: **60,000** images  
- Test set: **10,000** images  
- Classes:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

Source: https://github.com/zalandoresearch/fashion-mnist

---

## Project Structure


### References
Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion‑MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747.
TensorFlow (2024). Basic classification: Classify images of clothing.
https://www.tensorflow.org/tutorials/keras/classification
