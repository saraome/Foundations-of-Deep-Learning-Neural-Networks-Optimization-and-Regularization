### **Task 7 — Optimizer Comparison Challenge**

#### **1. Objective**
To compare the convergence speed and stability of SGD, SGD with Momentum, Adam, and AdamW optimizers, discuss how each navigates the loss landscape, and explain why adaptive optimizers like Adam often outperform classical methods.

#### **2. Results**

*   **SGD Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/6d4c7b5b-43ef-43b9-a65d-f0a6cb0a482c" />


*   **SGD with Momentum Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/4097707d-ad26-4c96-aee2-42ff470bdf81" />


*   **Adam Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/cf7ffbfa-906d-4c36-8a3f-d48fba9b6fdb" />


*   **AdamW Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/0b04a7ff-336f-4c11-b479-076493ea41a8" />



#### **3. Key Takeaway**
Adaptive optimizers like Adam and AdamW offer significant advantages in terms of convergence speed and robustness over classical optimizers (SGD, SGD with Momentum) by dynamically adjusting learning rates for each parameter, enabling more efficient navigation of complex loss landscapes and often leading to better performance and generalization with less hyperparameter tuning.
