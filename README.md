
---

## ⚙️ How It Works

1. The app provides a **drawing canvas** using `streamlit-drawable-canvas`.
2. The drawn image is captured and converted into a **grayscale 28×28 image**, just like the MNIST dataset format.
3. The image is then **normalized and inverted** (white digit on black background).
4. The pre-trained CNN model predicts the most likely digit.
5. The app displays:
   - 🧮 The **predicted digit**
   - 📊 A **bar chart** showing prediction confidence across all 10 digits.

---

## 🧩 Model Description

The trained model is a **Convolutional Neural Network (CNN)** optimized for recognizing small grayscale images.

### 🧱 Model Architecture Overview
- **Input Layer:** 28×28 grayscale images (MNIST format)
- **Conv2D Layers:** Extract important pixel patterns
- **MaxPooling2D Layers:** Reduce spatial dimensions
- **Flatten Layer:** Converts 2D matrices to 1D vector
- **Dense Layers:** Fully connected layers for classification
- **Output Layer:** 10 neurons with softmax activation (digits 0–9)

### 🏋️ Training Details
- **Dataset:** MNIST (60,000 training + 10,000 testing images)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 128
- **Epochs:** 10
- **Training Accuracy:** 98%
- **Test Accuracy:** 93.7%

> 🔍 Note: The model performs best when the drawn digit is **centered, thick, and similar in style to MNIST digits**.

---

## 🖥️ App Demo

When you run the Streamlit app, you will see an interface like this:



## 📂 Folder Structure

handwritten-digit-recognition/
│
├── appi.py # Main Streamlit application
├── handwritten_digit_model.h5 # Trained CNN model
├── requirements.txt # All required dependencies
├── .gitignore # Files/folders to ignore in Git
├── README.md # Project documentation
