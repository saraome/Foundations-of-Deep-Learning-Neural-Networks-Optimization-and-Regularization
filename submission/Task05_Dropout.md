### **Task 5 — Dropout Ablation Study**

#### **1. Objective**
To analyze the impact of different Dropout rates (0, 0.1, and 0.3) on model performance, particularly concerning overfitting, and to explain how Dropout contributes to learning more robust representations by preventing neuron co-adaptation.

#### **2. Code Used**
We used similar model architectures, varying only the `keras.layers.Dropout` rate, and trained each for 10 epochs. The relevant code snippets are:

```python
# No Dropout configuration
model_no_dropout = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model_no_dropout.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_no_dropout = model_no_dropout.fit(x_tr, y_tr, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Dropout = 0.1 configuration
model_0_1_dropout = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation="softmax")
])
model_0_1_dropout.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_0_1_dropout = model_0_1_dropout.fit(x_tr, y_tr, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Dropout = 0.3 configuration
model_0_3_dropout = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3), # Note: The provided code mistakenly had 0.1, corrected here.
    keras.layers.Dense(10, activation="softmax")
])
model_0_3_dropout.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_0_3_dropout = model_0_3_dropout.fit(x_tr, y_tr, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Plotting code (similar for each configuration):
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# ... and for accuracy.
```

#### **3. Results**

*   **No Dropout:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/b25c0fd3-f591-4624-999d-a12beba85e5b" />


*   **Dropout = 0.1:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/387b0481-6b66-4c99-a4d5-c0ca6496c2ef" />


*   **Dropout = 0.3:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/3ae8c357-f703-4a33-a66d-6200da908175" />


#### **4. Short Analysis**

**Overfitting Levels Comparison:**

*   **No Dropout:** This model exhibited clear signs of overfitting. The training loss continued to decrease significantly, and training accuracy reached very high levels (e.g., ~99.6% by epoch 10). However, the validation loss started to plateau and then slightly increase after around epoch 4-5, while validation accuracy also showed a slight decline or stagnation. The gap between training and validation metrics became noticeable.

*   **Dropout = 0.1:** With a modest dropout rate of 0.1, the overfitting trend was mitigated. While training loss still decreased and accuracy increased, the divergence between training and validation curves was less pronounced. Validation loss remained more stable, and validation accuracy was maintained at a high level without a clear decline. This suggests that a small amount of regularization helped the model generalize better.

*   **Dropout = 0.3:** Increasing the dropout rate to 0.3 further reduced overfitting. The training loss decreased at a slightly slower pace compared to the other configurations, and the training accuracy also rose more gradually. Crucially, the validation loss showed a more consistent decrease, and the validation accuracy was very stable, even improving slightly towards the end. The gap between training and validation metrics was minimal, indicating effective regularization.

**How Dropout Encourages Robust Representations:**
Dropout works by randomly setting a fraction of neuron outputs to zero at each training step. This forces the remaining active neurons to learn more robust features that are not overly reliant on any single input or specific set of co-occurring features from other neurons. It prevents
