### **Task 2 — Custom Image Generalization Test**

### **1. Objective**
To assess the model's generalization ability on a custom handwritten digit, analyze misclassification if it occurs, and discuss its relation to representation learning.

### **2. Code Used**
```python
import cv2

img_path = "/content/Untitled11.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28))
# Invert colors if background is white (MNIST style) - commented out as image is dark background
# resized = cv2.bitwise_not(resized)
normalized = resized / 255.0
input_image = normalized.reshape(1, 28, 28)

plt.imshow(resized, cmap="gray")
plt.title("Preprocessed Input")
plt.axis("off")
plt.show()

pred = model.predict(input_image)
print("Predicted Label:", np.argmax(pred))
print("True Label: 8") # Assuming the user intended to draw an '8'
print("Prediction probabilities:", pred)
```

### **3. Results**
*   **Custom Image:** The input image was a handwritten digit '8'.

<img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/61956947-cde7-4361-b84b-2eb1e94a78b2" />


*   **Predicted Label:** 2
*   **True Label (intended):** 8
*   **Prediction Probabilities:** `[[3.5310179e-05 1.1131167e-03 9.8010445e-01 7.4334680e-03 7.1340864e-06 1.4199516e-04 1.0751629e-02 9.7484635e-06 4.0314646e-04 3.1910385e-08]]` (The model assigned ~98% probability to '2').

### **4. Short Analysis**
The model **incorrectly classified the digit '8' as '2'**. Several factors could contribute to this misclassification:

*   **Distribution Shift:** The most likely reason is that the handwritten '8' differs significantly from the '8's the model was trained on in the MNIST dataset. The MNIST dataset typically features centered, relatively bold digits. the image, while clearly an '8' to a human, might have strokes, proportions, or a style that the model hasn't encountered enough during training.

*   **Noise and Preprocessing:** Although preprocessing steps like resizing and normalization were applied, subtle differences in how the digit was drawn (e.g., thickness of lines, gaps, overall shape) can act as 'noise' or 'unseen variations' for the trained model.

*   **Lack of Augmentation:** The model was trained on a relatively standard MNIST dataset. If it hadn't been exposed to augmented data (e.g., rotated, skewed, or slightly distorted digits) during training, its ability to generalize to vastly different handwritten styles diminishes.

### **5. Key Takeaway**
This misclassification highlights the challenge of **generalization** when there's a **distribution shift** between training data and real-world inputs. Models learn specific representations from their training data, and variations outside that learned distribution, even if semantically similar to humans, can lead to incorrect predictions. Effective models often require more diverse training data or robust augmentation strategies to handle such variations.
