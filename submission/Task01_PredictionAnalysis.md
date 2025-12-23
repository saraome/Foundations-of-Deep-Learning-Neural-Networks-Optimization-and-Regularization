
### **Task 1 — Deep Prediction Analysis**

### **1. Objective**
To explain why the model correctly predicted specific digit images by detailing the forward pass, the role of activation functions (ReLU and Softmax), and the influence of the Adam optimizer during training.

### **2. Code Used**
```python
# For sample index 14
sample_14 = x_test[14].reshape(1, 28, 28)
pred_14 = model.predict(sample_14)
print(f"Predicted Label (index 14): {np.argmax(pred_14)}")
print(f"True Label (index 14): {y_test[14]}")
plt.imshow(x_test[14], cmap="gray")
plt.show()

# For sample index 100
sample_100 = x_test[100].reshape(1, 28, 28)
pred_100 = model.predict(sample_100)
print(f"Predicted Label (index 100): {np.argmax(pred_100)}")
print(f"True Label (index 100): {y_test[100]}")
plt.imshow(x_test[100], cmap="gray")
plt.show()

# For sample index 2500
sample_2500 = x_test[2500].reshape(1, 28, 28)
pred_2500 = model.predict(sample_2500)
print(f"Predicted Label (index 2500): {np.argmax(pred_2500)}")
print(f"True Label (index 2500): {y_test[2500]}")
plt.imshow(x_test[2500], cmap="gray")
plt.show()
```

### **3. Results**
The model correctly predicted the labels for the selected samples:
*   **Sample 14:** Predicted Label: 1, True Label: 1

<img width="416" height="413" alt="image" src="https://github.com/user-attachments/assets/92d9374c-bd13-461e-8d39-033597adf131" />

 

*   **Sample 100:** Predicted Label: 6, True Label: 6

<img width="416" height="413" alt="image" src="https://github.com/user-attachments/assets/dc8a7066-061e-4470-93fc-898d6678388f" />


*   **Sample 2500:** Predicted Label: 2, True Label: 2

<img width="416" height="413" alt="image" src="https://github.com/user-attachments/assets/780aa063-55af-4c85-88b9-943309d14079" />


### **4. Short Analysis**

The model successfully predicted these digits due to how its parts work together:

**Forward Pass:** The 28x28 image is flattened into a single line of pixels. This line then goes through the first Dense layer (128 neurons), which learns important features. Then, it goes to the final Dense layer (10 neurons), which makes the actual predictions.

**Activation Functions:**

ReLU (in the hidden layer) adds non-linearity, allowing the model to learn complex patterns in the images (e.g., curves, lines that make up a digit).
Softmax (in the output layer) converts these learned patterns into probabilities for each digit (0-9). The digit with the highest probability is the model's prediction. For these correct predictions, Softmax gave a very high chance to the right digit.

**Adam Optimizer:** During training, the Adam optimizer adjusted the model's internal settings (weights and biases) very efficiently. It smartly fine-tuned the learning process, helping the model quickly find the best way to recognize different digits and generalize well to new images it hadn't seen before. This allowed it to achieve high accuracy and correctly classify these samples.

### **5. Key Takeaway**
The correct predictions highlight the model's successful generalization, which is attributed to the sequential data transformation through dense layers, the introduction of non-linearity via ReLU, clear probability outputs by Softmax, and efficient weight optimization by Adam, collectively enabling the model to learn and recognize digit patterns effectively from unseen data.
