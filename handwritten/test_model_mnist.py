import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the trained model
model = load_model(r"C:\Users\Sonali\Downloads\handwritten\handwritten_digit_model.h5")

# Compile the model to enable metrics evaluation
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess the test data
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Evaluate accuracy on the first 1000 test samples
loss, acc = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
print(f"✅ Accuracy on first 1000 MNIST test samples: {acc}")
