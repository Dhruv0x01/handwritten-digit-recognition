from tensorflow.keras.datasets import mnist   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.optimizers import Adam

from utils.preprocess import normalize_images, reshape_images

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess images

#Normalized
x_train = normalize_images(x_train)
x_test = normalize_images(x_test)

#Reshaped
x_train = reshape_images(x_train)
x_test = reshape_images(x_test)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# CNN Architecture 
model = Sequential()


model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
# Model is Defined, now time to compile model

model.compile(
    optimizer=Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)




