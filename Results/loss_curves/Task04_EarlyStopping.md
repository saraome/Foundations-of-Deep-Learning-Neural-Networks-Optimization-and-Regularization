### **Task 4 — EarlyStopping Behavior Analysis**

### **1. Objective**
To analyze how `keras.callbacks.EarlyStopping` works by observing its behavior with different optimizers (Adam and SGD), and to understand its role as a regularization technique.

### **2. Code Used**
```python
# Adam Optimizer Experiment
model_es = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model_es.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ear_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history_es = model_es.fit(
    x_tr, y_tr,
    epochs=50, # Set a high number, ES will stop it
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[ear_stopping],
    verbose=0 # Suppress verbose output for brevity
)

# SGD Optimizer Experiment
model_sgd_es = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model_sgd_es.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ear_stopping_sgd = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history_sgd_es = model_sgd_es.fit(
    x_tr, y_tr,
    epochs=50, # Set a high number, ES will stop it
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[ear_stopping_sgd],
    verbose=0 # Suppress verbose output for brevity
)
```

### **3. Results**
*   **Adam Optimizer (with `patience=3`, `restore_best_weights=True`):**
    *   Training stopped at **Epoch 7**. The best validation loss was observed at Epoch 4 (approx. 0.0726).
    *   Plots:
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/45c2fc66-6ad2-4ba2-b0a1-00aad3c8f628" />


*   **SGD Optimizer (with `patience=3`, `restore_best_weights=True`):**
    *   Training completed all **50 epochs**. EarlyStopping was not triggered within this period.
    *   Plots:
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/9f1a4aed-b4df-4070-a3ac-e41d2a86338c" />


### **4. Short Analysis**
1.  **At which epoch did training stop?**
    *   **Adam Optimizer:** Training stopped at **Epoch 7**. This means that after Epoch 4 (where the best validation loss was achieved), the validation loss did not improve for 3 consecutive epochs (Epochs 5, 6, and 7), so EarlyStopping halted the training. The model was then reverted to the weights from Epoch 4.
    *   **SGD Optimizer:** Training ran for all **50 epochs**. EarlyStopping was *not* triggered because the validation loss did not consistently worsen for 3 consecutive epochs. This suggests a more gradual or stable convergence with SGD, or fluctuations that did not meet the stopping criteria within the given epoch limit.

2.  **Why does the validation loss control this decision?**
    *   Validation loss is used because it measures the model's performance on unseen data. The goal is to build a model that generalizes well, not just memorizes the training data. If the training loss continues to decrease but validation loss stops improving or starts to increase, it indicates the model is beginning to overfit.

3.  **What happens if you increase patience (e.g., to 5)?**
    *   Increasing `patience` to 5 means the model would wait for 5 non-improving validation loss epochs before stopping. This would generally lead to training for more epochs. While it might sometimes help the model overcome minor fluctuations and find a slightly better minimum, it also increases the risk of more severe overfitting by allowing the model to train longer past its optimal generalization point.

4.  **Would a different optimizer (e.g., SGD) change the EarlyStopping pattern?**
    *   Yes, as shown in the results, a different optimizer significantly changes the EarlyStopping pattern. Adam, being an adaptive optimizer, tends to converge faster and more aggressively, often leading to a quicker drop in validation loss and a clearer optimal point, thus triggering EarlyStopping earlier. SGD, with its simpler update rule and potentially slower, more oscillatory convergence, might not trigger EarlyStopping as readily, especially if validation loss fluctuates or improves very gradually over many epochs.

5.  **Explain how EarlyStopping acts as an indirect form of regularization.**
    *   EarlyStopping prevents overfitting, which is the core goal of regularization. By stopping training when the model's performance on unseen validation data begins to degrade, it prevents the model from continuing to learn noise and specific details of the training set. This ensures that the model maintains a good balance between fitting the training data and generalizing to new data, effectively acting as an indirect regularization technique.

### **5. Key Takeaway**
EarlyStopping is a crucial callback for preventing overfitting by intelligently halting training based on validation performance, ensuring the model generalizes well, and its effectiveness can vary significantly with the choice of optimizer and patience.
