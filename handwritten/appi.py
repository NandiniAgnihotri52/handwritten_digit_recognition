import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Loading the trained model
model = load_model(r"C:\Users\Sonali\Downloads\handwritten\handwritten_digit_model.h5")

st.title("🖊️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below and click **Predict** to see the result!")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # using the threshold to remove noise
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        # Now finding bounding box of the drawn digit
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit = img[y:y+h, x:x+w]

            # Resizing the digit region to 20x20
            digit = cv2.resize(digit, (20, 20))

            # Create blank 28x28 image and paste digit centered
            new_img = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - 20) // 2
            y_offset = (28 - 20) // 2
            new_img[y_offset:y_offset+20, x_offset:x_offset+20] = digit
        else:
            new_img = img

        # Normalizing
        new_img = new_img / 255.0
        new_img = (new_img > 0.2).astype(float)
        new_img = new_img.reshape(1, 28, 28, 1)

        st.write("🧩 Image Stats:")
        st.json({
            "shape": new_img.shape[1:],
            "min": float(new_img.min()),
            "max": float(new_img.max()),
            "mean": float(new_img.mean()),
        })

        # Predicting
        prediction = model.predict(new_img)
        result = np.argmax(prediction)

        st.success(f"🧮 Predicted Digit: **{result}**")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit before clicking Predict.")
