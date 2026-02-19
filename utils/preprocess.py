import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Normalize pixel values (0-255 -> 0-1)
def normalize_images(images):
    return images.astype("float32") / 255.0

# Reshape images for CNN (add channel dimension)
def reshape_images(images):
    return images.reshape(images.shape[0], 28, 28, 1)

# Preprocess single user-drawn image (for GUI)
def preprocess_user_image(image):
    #Resize to 28x28
    image = image.resize((28,28)) 

    #Convert to grayscale
    image = image.convert("L")      

    image_array = np.array(image)

    # Light Noise reduction (Gaussian blur)
    image_array = cv2.GaussianBlur(image_array, (3, 3), 0)

    # Threshold to make digit clearer (binary style like MNIST)
    ret, thresh = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_array = thresh


    # Find bounding box of digit
    coords = np.column_stack(np.where(image_array > 0))

    if coords.shape[0] > 0:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        digit = image_array[x_min:x_max, y_min:y_max]

        #Resize digit to 20x20 (MNIST style)
        digit = cv2.resize(digit, (20, 20))

        #Place it in center of 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = digit
        image_array = canvas

    # Scale
    image_array = image_array.astype("float32") / 255.0

    # Reshape for model
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array