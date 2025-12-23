### **Task 3 — Consolidated Epoch-Based Learning Curve Analysis**

#### **1. Objective**
To analyze how increasing the number of training epochs impacts model performance, identify signs of overfitting, and understand the influence of the Adam optimizer on convergence speed and stability.

#### **2. Experiments & Observations**
We conducted three experiments, training the model for 5, 10, and 20 epochs respectively, and observed the following:

*   **5 Epochs:**
    *   **Learning Curves:** Both training and validation loss decreased, and accuracy increased steadily. The validation loss was still improving.
    *   **Overfitting:** No significant signs of overfitting were observed. The model was likely *underfitting* slightly, as it still had room to learn.
    *   **Adam Optimizer:** Adam demonstrated fast initial convergence, quickly reducing loss and improving accuracy. Its adaptive learning rate helped the model efficiently find a good path in the early stages of training.
    
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/d7efb0a4-1668-4951-b16d-8e481f5fe565" /> 


*   **10 Epochs:**
    *   **Learning Curves:** Training loss continued to decrease, and training accuracy continued to rise. However, the **validation loss started to plateau and then slightly increase** after around 5-6 epochs, while validation accuracy also plateaued.
    *   **Overfitting:** Subtle signs of overfitting became apparent. The model started to perform slightly worse on unseen validation data, indicating it was beginning to memorize training data specifics rather than generalizing.
    *   **Adam Optimizer:** Adam continued to minimize training loss effectively. However, without additional regularization, its efficiency in reducing training error began to expose the model's tendency to overfit the training data.
    
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/e678a829-c231-46dc-bb4a-ace36fcad443" />


*   **20 Epochs:**
    *   **Learning Curves:** Training loss continued its downward trend, and training accuracy reached very high levels (approaching 100%). In contrast, the **validation loss significantly increased** after reaching a minimum around 5-8 epochs, and validation accuracy showed a clear decline or stagnation after its peak.
    *   **Overfitting:** Clear and significant overfitting was observed. The large gap between training and validation metrics indicates that the model had become highly specialized to the training data, losing its ability to generalize to new data.
    *   **Adam Optimizer:** While Adam successfully drove down the training error to very low values, it also facilitated the overfitting process by continuing to adjust weights to fit the training data more perfectly. This highlights that Adam, while efficient, doesn't inherently prevent overfitting when the model is trained for too long without regularization.

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/c5113b9c-33f8-43e2-9aeb-306be10efb52" />


#### **3. Key Takeaway**
Choosing the right number of epochs is crucial for optimal model performance. Training for too few epochs can lead to underfitting (model hasn't learned enough), while training for too many epochs can lead to **overfitting** (model memorizes training data and performs poorly on new data). The Adam optimizer accelerates learning, but its effectiveness also means it can quickly lead to overfitting if not managed with techniques like early stopping or regularization. The ideal stopping point is often found when validation loss reaches its minimum before starting to rise.
