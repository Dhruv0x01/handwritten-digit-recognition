import numpy as np
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
    image = image.resize((28,28))   #Resize
    image = image.convert("L")      #Convert to grayscale
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array