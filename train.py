from tensorflow.keras.datasets import mnist   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

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

# Block 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

# Block 2
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

# Block 3
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()
# Model is Defined, now time to compile model

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)


# Train the Model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# -------------------------------
# Misclassfication Analysis
# -------------------------------

# Predict on test set
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Find misclassified indices
misclassified = np.where(predicted_labels != y_test)[0]
print("Total misclassified:", len(misclassified))

plt.figure(figsize=(8,8))

for i in range(9):
    index = misclassified[i]
    confidence = np.max(predictions[index])

    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[index].reshape(28,28), cmap='gray')
    plt.title(
        f"True: {y_test[index]}\nPred: {predicted_labels[index]} ({confidence:.2f})"
    )
    plt.axis("off")

plt.tight_layout
plt.show()


# Save the model
model.save("models/digit_model.keras")






