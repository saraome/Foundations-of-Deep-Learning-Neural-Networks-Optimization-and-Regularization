### **Task 4 — EarlyStopping Behavior Analysis**
### **Results**
*   **Adam Optimizer (with `patience=3`, `restore_best_weights=True`):**
    *   Training stopped at **Epoch 7**. The best validation loss was observed at Epoch 4 (approx. 0.0726).
    *   Plots:
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/45c2fc66-6ad2-4ba2-b0a1-00aad3c8f628" />


*   **SGD Optimizer (with `patience=3`, `restore_best_weights=True`):**
    *   Training completed all **50 epochs**. EarlyStopping was not triggered within this period.
    *   Plots:
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/9f1a4aed-b4df-4070-a3ac-e41d2a86338c" />

