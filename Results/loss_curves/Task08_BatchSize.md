### **Task 8 — Batch Size & Gradient Noise Experiment**

#### **1. Objective**
To investigate how different batch sizes (8, 32, 128) influence model training, specifically focusing on gradient noise, convergence speed, generalization, and the smoothness of learning curves.

#### **2. Code Used**
For each batch size, a Keras Sequential model with `Flatten`, `Dense(128, activation="relu")`, and `Dense(10, activation="softmax")` layers was trained for 10 epochs using the Adam optimizer.

```python
# Example for batch_size = 8
model_adam = keras.Sequential([...])
model_adam.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_model_adam = model_adam.fit(x_tr, y_tr, epochs=10, batch_size=8, validation_data=(x_val, y_val))
# Similar code structure for batch_size = 32 and 128.
```

#### **3. Results**

*   **Batch Size = 8:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/ca8b3abd-7419-4b2d-a69e-ae2bdf922428" />


*   **Batch Size = 32:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/00b46c9f-b464-4543-9029-cf9f11b237ce" />


*   **Batch Size = 128:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/7578f570-8bef-482c-b996-c6a6e9fc7a20" />


#### **4. Short Analysis**

*   **Gradient Noise:** Smaller batch sizes (like 8) introduce more gradient noise, causing choppier loss curves. Larger batch sizes (like 128) have smoother curves due to less noise.
*   **Generalization:** Smaller batches (8, 32) can sometimes lead to better generalization by helping the model escape sharp local minima (due to noise). Larger batches (128) might converge to flatter, less optimal minima, potentially leading to worse generalization if trained for too long.
*   **Convergence Speed:** Larger batch sizes (128) often show faster progress per epoch because fewer updates are needed. However, smaller batches can achieve a good solution faster in terms of total training time, as each update is more frequent and diverse.
*   **Loss Curve Smoothness:** As batch size increases, the training and validation loss curves become noticeably smoother, as seen from batch 8 (noisy) to batch 128 (smoother).

#### **5. Key Takeaway**
Batch size significantly impacts training dynamics. Smaller batch sizes introduce more gradient noise which can aid in finding better generalizing solutions but lead to noisy loss curves. Larger batch sizes provide smoother training and faster per-epoch convergence but might settle for suboptimal solutions. A moderate batch size (like 32) often strikes a good balance.
