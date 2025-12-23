### **Task 10 — Weight Inspection & Model Capacity Analysis**

#### **1. Objective**
To analyze the implications of the number of parameters in the first Dense layer on model capacity and overfitting, and to explain how various regularization techniques (Dropout, L2, EarlyStopping) mitigate this risk.

#### **2. Code Used**
```python
# Extraction of weights from the first Dense layer of model_GELU
first_dense_layer = model_GELU.layers[1]
w, b = first_dense_layer.get_weights()

print("Shape of kernel weights (w) from the first Dense layer:", w.shape)
print("Shape of bias (b) from the first Dense layer:", b.shape)
```

#### **3. Results**
*   Shape of kernel weights (w) from the first Dense layer: `(784, 128)`
*   Shape of bias (b) from the first Dense layer: `(128,)`

#### **4. Short Analysis**

*   **Why the number of parameters is so large:**
    The input images are 28x28 pixels. When flattened, this becomes a vector of 784 features. The first Dense layer has 128 neurons. Therefore, each of the 128 neurons connects to all 784 input features. This results in `784 * 128 = 100,352` weights for the connections (kernel weights). Additionally, each of the 128 neurons has its own bias term, adding `128` bias parameters. In total, this single layer has `100,352 + 128 = 100,480` parameters. This number is considered large for a single hidden layer in a simple neural network, indicating high model capacity.

*   **How high model capacity increases overfitting risk:**
    High model capacity means the model has a very large number of parameters relative to the size and complexity of the training data. Such models have many degrees of freedom and can easily 'memorize' the training data, including noise and irrelevant patterns, rather than learning generalizable features. When a model memorizes the training data, its performance on that data will be excellent (low training loss, high training accuracy), but it will struggle to perform well on new, unseen data (high validation loss, low validation accuracy), which is the definition of overfitting.

*   **How techniques like Dropout, L2, and EarlyStopping each mitigate this risk differently:**
    *   **Dropout:** Acts by randomly deactivating a fraction of neurons during each training step. This forces the network to learn more robust features that are not overly reliant on any single neuron or specific set of co-adapting neurons. It effectively trains an ensemble of smaller networks, reducing the inter-dependency of neurons and improving generalization.
    *   **L2 Regularization (Weight Decay):** Adds a penalty to the loss function proportional to the square of the magnitude of the weights. This encourages the model to use smaller weights, making the decision boundary smoother and less sensitive to individual data points. Smaller weights reduce the complexity of the model, thus making it less prone to memorizing the training data's noise.
    *   **EarlyStopping:** This is a meta-regularization technique that monitors the model's performance on a validation set during training. It stops the training process once the validation performance (e.g., validation loss) stops improving for a specified number of epochs (patience). This prevents the model from continuing to train past the point where it starts to overfit, saving the weights from the epoch with the best validation performance. It's an efficient way to prevent models from learning the noise in the training data.

#### **5. Key Takeaway**
The large number of parameters in a neural network contributes to high model capacity, increasing the risk of overfitting. Regularization techniques like Dropout, L2 regularization, and EarlyStopping are crucial tools that combat overfitting by either simplifying the model (L2), making it more robust (Dropout), or halting training before performance degrades on unseen data (EarlyStopping).
