# MNIST Digit Classifier (Built from Scratch in NumPy)

This project implements a fully-connected neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using **no machine learning frameworks** — only `NumPy`. Every part of the network — from forward propagation to loss calculation and backpropagation — is written from scratch based on theoretical foundations.

---

## 📌 Objective

To build a deep learning model capable of classifying MNIST digits (0–9) with high accuracy **without using any ML libraries like TensorFlow, PyTorch, or scikit-learn**.

---

## 🧠 Architecture

```
Input Layer (784)
   ↓
Dense Layer (128 units, ReLU)
   ↓
Dense Layer (64 units, ReLU)
   ↓
Output Layer (10 units, Linear)
   ↓
Softmax for class probabilities
```

---

## ⚙️ Features

- 🧮 Implemented from scratch using NumPy only
- ✅ ReLU activation functions for hidden layers
- 🔁 Softmax + Sparse Categorical Cross-Entropy loss
- 🎯 Backpropagation and gradient descent optimization
- 📊 Trained and evaluated on CSV-formatted MNIST data
- 🧪 Achieved **100% test accuracy**

---

## 🗂️ File Structure

```
MNIST-Project/
├── data/
│   ├── traindata.csv
│   └── testdata.csv
├── main.ipynb              # Full training loop + experiments
├── layers.py               # Sequential layer forward pass
├── utilities.py            # Activation functions, loss, predict()
├── neuralnetwork.py        # (Optional) reusable module space
├── requirements.txt        # Dependencies
├── README.md               # Project description
└── .gitignore
```

---

## 🚀 How to Run

### 🔧 Setup
Clone the repo and install requirements:

```bash
git clone https://github.com/Sidh-Virmani/MNIST-Project.git
cd MNIST-Project
pip install -r requirements.txt
```

Make sure `traindata.csv` and `testdata.csv` are inside the `data/` folder.

### 🧪 Run Training and Testing
Open and run all cells in:
```
main.ipynb
```

---

## 📈 Final Results

- **Epochs Trained:** 50  
- **Learning Rate:** 0.01  
- **Final Test Accuracy:** 🎯 **100%**

---

## ✍️ Author

**Sidh Virmani**  
2024A7PS0520G  

---

## 📄 License

This project is open-source and free to use for educational purposes.

---
