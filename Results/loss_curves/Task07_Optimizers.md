### **Task 7 — Optimizer Comparison Challenge**

#### **1. Objective**
To compare the convergence speed and stability of SGD, SGD with Momentum, Adam, and AdamW optimizers, discuss how each navigates the loss landscape, and explain why adaptive optimizers like Adam often outperform classical methods.

#### **2. Code Used**
For each optimizer, a Keras Sequential model with `Flatten`, `Dense(128, activation="relu")`, and `Dense(10, activation="softmax")` layers was trained for 10 epochs. Each model was compiled with its respective optimizer configuration:

*   **SGD:** `keras.optimizers.SGD(learning_rate=0.01)`
*   **SGD with Momentum:** `keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)`
*   **Adam:** `keras.optimizers.Adam(learning_rate=0.001)` (default learning rate for Adam)
*   **AdamW:** `keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001)` (default learning rate and a typical weight decay)

```python
# Example for SGD
model_sgd = keras.Sequential([...])
model_sgd.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), ...)
history_sgd = model_sgd.fit(x_tr, y_tr, epochs=10, validation_data=(x_val, y_val))
# Similar code structure for other optimizers.
```

#### **3. Results**

*   **SGD Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/6d4c7b5b-43ef-43b9-a65d-f0a6cb0a482c" />


*   **SGD with Momentum Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/4097707d-ad26-4c96-aee2-42ff470bdf81" />


*   **Adam Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/cf7ffbfa-906d-4c36-8a3f-d48fba9b6fdb" />


*   **AdamW Optimizer:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/0b04a7ff-336f-4c11-b479-076493ea41a8" />


#### **4. Short Analysis**

**Convergence Speed and Stability:**

*   **SGD:** Showed the slowest convergence among all optimizers. Both training and validation loss decreased gradually, and accuracy increased steadily but slowly. The curves were relatively smooth, indicating stable but slow progress.
*   **SGD with Momentum:** Significantly faster convergence than plain SGD. The momentum term helped accelerate the learning process, leading to a quicker drop in loss and rise in accuracy. The curves were generally smoother than plain SGD, as momentum helps dampen oscillations.
*   **Adam:** Demonstrated the fastest convergence, reaching high accuracy and low loss within a few epochs. The curves show rapid initial improvement and then stable progress. It consistently achieved better performance metrics in fewer epochs compared to SGD variants.
*   **AdamW:** Similar to Adam, AdamW also showed fast convergence. Its performance was very close to Adam, often slightly outperforming in terms of validation loss/accuracy or showing better generalization in the long run due to its decoupled weight decay.

**How Each Optimizer Navigates the Loss Landscape:**

*   **SGD:** Takes small, consistent steps in the direction opposite to the gradient. In a complex, non-convex loss landscape, this can lead to slow progress through flat regions or getting stuck in local minima. Its steps are uniform across all dimensions, regardless of the steepness of the gradient in those dimensions.
*   **SGD with Momentum:** Introduces a "velocity" term that accumulates past gradients. This helps the optimizer build speed in consistent directions and "roll" over small obstacles (shallow local minima) in the loss landscape. It also smooths out oscillations in directions with high curvature, allowing for larger effective steps.
*   **Adam:** Combines the benefits of momentum and RMSprop. It computes adaptive learning rates for each parameter. It estimates both the first moment (mean) and the second moment (uncentered variance) of the gradients. This allows it to adapt the step size for each parameter individually, taking larger steps in dimensions with small, consistent gradients and smaller steps in dimensions with large or oscillating gradients. This adaptive nature makes it very efficient at navigating complex, high-dimensional loss landscapes, including those with saddle points or narrow valleys.
*   **AdamW:** A variant of Adam that decouples weight decay from the adaptive learning rate update. In original Adam, weight decay (L2 regularization) is mixed with the adaptive gradient scaling, which can sometimes lead to suboptimal results. AdamW applies weight decay directly to the weights, rather than through the gradients, often leading to better regularization and improved generalization, especially in larger models or when training for many epochs.

**Why Adam Often Outperforms Classical Optimizers:**
Adam (and its variants like AdamW) often outperforms classical optimizers like SGD and SGD with Momentum due to its adaptive learning rate mechanism.
1.  **Adaptive Learning Rates:** Adam dynamically adjusts the learning rate for each parameter. This means it can effectively handle sparse gradients (e.g., in NLP tasks or deep networks with ReLU activations) and noisy gradients, accelerating convergence significantly. Classical optimizers use a single learning rate for all parameters, making them sensitive to hyperparameter tuning and potentially slow in certain dimensions.
2.  **Robustness to Initial Conditions:** Adam is generally less sensitive to the initial choice of learning rate and other hyperparameters compared to SGD, making it easier to use out-of-the-box.
3.  **Efficiency:** It requires less memory and is computationally efficient, making it practical for large datasets and complex models.
4.  **Handling Sparse Gradients:** Its adaptive nature allows it to make significant progress even when gradients are sparse or very small, which is common in deep learning.

#### **5. Key Takeaway**
Adaptive optimizers like Adam and AdamW offer significant advantages in terms of convergence speed and robustness over classical optimizers (SGD, SGD with Momentum) by dynamically adjusting learning rates for each parameter, enabling more efficient navigation of complex loss landscapes and often leading to better performance and generalization with less hyperparameter tuning.
