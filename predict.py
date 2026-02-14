from tensorflow.keras.models import load_model 
import numpy as np

# Load trained model
model = load_model("models/digit_model.keras")

def predict_digit(image_array):
    prediction = model.predict(image_array)
    return np.argmax(prediction)
