### **Task 9 — Activation Function Swap (ReLU vs Tanh vs GELU)**

#### **1. Objective**
To analyze how different activation functions (Tanh, Softsign, and GELU, compared to the default ReLU) affect the model's performance, gradient flow, and identify scenarios where specific activations are more advantageous.

#### **2. Code Used**
For each activation function, a Keras Sequential model with `Flatten`, `Dense(128, activation="[ACTIVATION_FUNCTION]")`, and `Dense(10, activation="softmax")` layers was trained for 10 epochs using the Adam optimizer. The activation function string was replaced accordingly (`"tanh"`, `"softsign"`, `"gelu"`).

```python
# Example for Tanh
model_Tanh = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="tanh"),
    keras.layers.Dense(10, activation="softmax")
])
model_Tanh.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_model_Tanh = model_Tanh.fit(x_tr, y_tr, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Similar code structure for Softsign and GELU.
```

#### **3. Results**

*   **Tanh Activation:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/b546e4b2-e13c-4927-97c5-a9ea233dfd25" />


*   **Softsign Activation:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/d7387ccb-1cb4-43f7-ad7f-3d91bc50dc8b" />


*   **GELU Activation:**
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/19c973e5-30c4-48e1-a36c-ee8776d7df38" />


#### **4. Short Analysis**

*   **How each activation affects gradient flow:**
    *   **Tanh (Hyperbolic Tangent):** This activation function outputs values between -1 and 1. It is a zero-centered function, which can help in accelerating convergence compared to sigmoid, as gradients are less likely to oscillate. However, like sigmoid, it still suffers from the vanishing gradient problem for very large or very small inputs, where the gradient approaches zero, slowing down learning.
    *   **Softsign:** Similar to Tanh, Softsign is also S-shaped and outputs values between -1 and 1. It is defined as `x / (1 + abs(x))`. Its derivative is `1 / (1 + abs(x))^2`, which decays polynomially rather than exponentially like Tanh. This means its gradients vanish slower than Tanh, potentially allowing for better gradient flow in very deep networks compared to Tanh, though still slower than ReLU for positive values.
    *   **GELU (Gaussian Error Linear Unit):** This is a non-monotonic activation function that combines properties of ReLU and dropout. It smooths out ReLU and is defined as `x * Φ(x)`, where `Φ(x)` is the cumulative distribution function of the standard Gaussian distribution. GELU effectively acts as a ReLU with a 'probabilistic' switch, where the probability of activation depends on the input value. Its non-monotonic nature and smoother derivative help in addressing vanishing gradients and lead to more stable training.

*   **Which activations risk vanishing gradients:**
    *   Both **Tanh** and **Softsign** risk vanishing gradients, especially for inputs far from zero. Tanh's gradients vanish faster due to its exponential decay. Softsign's polynomial decay makes it slightly more resistant but still susceptible.
    *   **ReLU** (which was the default in previous tasks) mitigates the vanishing gradient problem for positive inputs by having a constant gradient of 1. For negative inputs, the gradient is 0, leading to the 
