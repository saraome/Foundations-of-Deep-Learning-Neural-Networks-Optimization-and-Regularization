### **Task 6 — L2 Regularization Experiment**

#### **1. Objective**
To examine how different strengths of L2 regularization (0.0001, 0.001, and 0.01) affect the model's learning, its tendency to overfit, and the general trend of validation loss, as well as to explain L2's role in weight management.

#### **2. Code Used**
For each regularization strength, a separate Keras Sequential model was defined and trained for 10 epochs. Each model included `keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(VALUE))`, where `VALUE` was varied. The models were compiled with `adam` optimizer and `sparse_categorical_crossentropy` loss.

```python
# Example for L2 = 0.0001
model_l2_0001 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(10, activation="softmax")
])
model_l2_0001.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_l2_0001 = model_l2_0001.fit(x_tr, y_tr, epochs=10, batch_size=32, validation_data=(x_val, y_val))
# Plotting code was similar for all three experiments.
```

#### **3. Results**

*   **L2 Regularization = 0.0001:** (Light Regularization)
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/fe851f8e-49e2-4ea2-b33c-d92117dfeadb" />


*   **L2 Regularization = 0.001:** (Moderate Regularization)
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/7584837d-bb17-4276-8ce9-2ebde4644cf6" />


*   **L2 Regularization = 0.01:** (Strong Regularization)
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/96b4814c-7e29-43fa-b211-1b92b0647796" />


#### **4. Short Analysis**

**How L2 Reduces Weight Magnitude:** L2 regularization adds a penalty to the loss function that is proportional to the square of the magnitude of the model's weights. This penalty encourages the model to use smaller weights. When weights are small, it means that no single feature (or pixel in our case) has too much influence on the prediction, making the model less sensitive to small changes in the input data. This process effectively 'shrinks' the weights towards zero, but unlike L1 regularization, it doesn't force them to be exactly zero.

**Why Smaller Weights Improve Generalization:** Models with very large weights can easily learn to perfectly fit the training data, including its noise, leading to overfitting. Smaller weights help make the model simpler and smoother, reducing its ability to memorize specific training examples. This leads to better *generalization*, meaning the model performs well on new, unseen data, not just the data it was trained on.

**How L2 Changes the Validation Loss Trend:**

*   **L2 = 0.0001 (Light):** With a small L2 penalty, the model still showed some signs of overfitting, similar to having no regularization. The training loss was lower than validation loss, and validation accuracy plateaued or slightly decreased towards the end, showing it wasn't strong enough to completely prevent overfitting.

*   **L2 = 0.001 (Moderate):** This value provided a good balance. The gap between training and validation loss was smaller, and validation accuracy remained stable or slightly improved over the 10 epochs. The model generalized better, as the penalty successfully pushed weights to be smaller without severely impacting the model's ability to learn important features.

*   **L2 = 0.01 (Strong):** With a much larger L2 penalty, the model's learning was significantly hindered. Both training and validation losses were higher overall, and accuracies were lower. This indicates *underfitting*, as the strong penalty forced the weights to be too small, preventing the model from learning the complex patterns necessary to accurately classify digits.

#### **5. Key Takeaway**
L2 regularization is an effective technique to prevent overfitting by encouraging smaller weights. The choice of the regularization strength (the L2 coefficient) is crucial: too little and the model still overfits; too much and it underfits. The goal is to find a moderate value that helps the model generalize well to new data.
