# Foundations of Deep Learning  
**Neural Networks, Optimization, and Regularization**

## 📘 Project Overview
This project is a structured, hands-on learning notebook that introduces the **foundations of deep learning** using **TensorFlow / Keras**.  
It combines theory, coding, visualization, and experimental tasks to explain how neural networks learn, optimize, and generalize.

The experiments are based on the **MNIST handwritten digit dataset** and progress from a simple neural network to advanced concepts such as regularization, early stopping, and optimizer comparison.

---

## 📂 Repository Structure
project/  
├── notebook.ipynb  
├── README.md  
├── results/  
│   ├── predictions/  
│   ├── loss_curves/  
│   └── optimizer_tests/  
└── submission/  
    ├── Task01_PredictionAnalysis.md  
    ├── Task02_CustomDigit.md  
    ├── Task03_Epochs.md  
    ├── Task04_EarlyStopping.md  
    ├── Task05_Dropout.md  
    ├── Task06_L2.md  
    ├── Task07_Optimizers.md  
    ├── Task08_BatchSize.md  
    ├── Task09_Activations.md  
    └── Task10_Weights.md  

---

## 🧠 Notebook Sections

### Section 1 — Building & Training a Neural Network
- Load and normalize MNIST data  
- Build a fully connected neural network  
- ReLU and Softmax activations  
- Train using the Adam optimizer  
- Visualize predictions and training curves  

**Goal:**  
Understand forward pass, loss computation, and basic model behavior.

---

### Section 2 — Regularization & Training Control
- Proper dataset splitting (train / validation / test)  
- Dropout regularization  
- L2 (weight decay) regularization  
- EarlyStopping callback  
- Final evaluation on unseen test data  

**Goal:**  
Learn how to reduce overfitting and improve model generalization.

---

### Section 3 — Student Tasks & Experiments
- Prediction behavior analysis  
- Custom handwritten digit generalization test  
- Epoch comparison (5, 10, 20 epochs)  
- Dropout ablation study  
- L2 regularization tuning  
- Optimizer comparison (SGD, Momentum, Adam, AdamW)  
- Batch size experiments  
- Activation function comparison  
- Weight inspection and model capacity analysis  

**Goal:**  
Build intuition about optimization dynamics, regularization strength, and architectural choices.

---

## ⚙️ Requirements
pip install tensorflow numpy matplotlib opencv-python

---

## ▶️ How to Run
1. Clone the repository  
git clone <your-repository-url>  

2. Navigate to the project directory  
cd project  

3. Open the notebook  
jupyter notebook notebook.ipynb  

---

## 📊 Results
- Training & validation loss/accuracy plots  
- Prediction visualizations  
- Optimizer comparison curves  
- Regularization impact analysis  

All outputs should be saved inside the `results/` directory.

---

## 🎯 Learning Outcomes
By completing this project, you will:
- Understand how neural networks learn representations  
- Analyze training and validation curves  
- Compare optimizers and regularization methods  
- Identify overfitting and underfitting  
- Gain practical intuition for deep learning workflows  

---

## ✅ Submission Notes
- The notebook must run without errors  
- Follow the required folder structure  
- Each task must be documented in Markdown  
- Keep explanations concise and clear  

---

## 📌 Final Note
This project focuses on **understanding model behavior**, not just achieving high accuracy.  
Clear explanations and well-organized experiments are key.
