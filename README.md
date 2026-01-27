# Fashion‑MNIST Image Classification
## CNN vs Random Forest Comparison

## Overview
This project evaluates two machine learning approaches for classifying fashion products using the **Fashion‑MNIST dataset**:
- A **Convolutional Neural Network (CNN)**
- A **Random Forest classifier**

The goal is to compare their performance in terms of accuracy, precision, recall, training time, and robustness to visually similar clothing categories.

---

## Dataset
The Fashion‑MNIST dataset contains **70,000 grayscale images** of fashion products, each with a resolution of **28×28 pixels**, distributed across **10 classes**.

- Training set: 60,000 images  
- Test set: 10,000 images  
- Classes:
  - T‑shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

Dataset source:  
https://github.com/zalandoresearch/fashion-mnist

---

## Project Structure
├── Fashion‑MNIST Classification Project.ipynb # Jupyter notebook with full implementation <br>
├── data/ # Fashion‑MNIST .gz files <br>
│ ├── train-images-idx3-ubyte.gz <br>
│ ├── train-labels-idx1-ubyte.gz <br>
│ ├── t10k-images-idx3-ubyte.gz <br>
│ └── t10k-labels-idx1-ubyte.gz <br>
├── README.md # Project documentation


---

## Methods

### Convolutional Neural Network (CNN)
- Two convolutional layers with ReLU activation
- Max‑pooling layers for spatial down‑sampling
- Fully connected dense layer with dropout
- Softmax output layer for 10‑class classification
- Optimizer: Adam
- Loss function: Sparse categorical cross‑entropy

### Random Forest Classifier
- Images flattened into 784‑dimensional feature vectors
- Ensemble of 200 decision trees
- No spatial feature learning
- Trained using default Gini impurity criterion

---

## Results

| Model | Test Accuracy | Training Time |
|------|--------------|---------------|
| CNN | **90.65%** | ~253 seconds |
| Random Forest | 87.79% | ~441 seconds |

Key observations:
- The CNN outperforms the Random Forest across all evaluation metrics
- Visually similar classes (e.g., Shirt vs T‑shirt) are challenging for both models
- The Random Forest struggles due to lack of spatial awareness

---

## Conclusion
The Convolutional Neural Network demonstrates superior performance for image‑based classification tasks, achieving higher accuracy and better generalization than the Random Forest classifier. Despite its simpler implementation, the Random Forest is less effective for complex visual data.

**Recommendation:** The CNN is better suited for production‑level fashion image classification systems.

---

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit‑learn

Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### References
Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion‑MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747.
TensorFlow (2024). Basic classification: Classify images of clothing.
https://www.tensorflow.org/tutorials/keras/classification
