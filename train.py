from tensorflow.keras.datasets import mnist   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
import time 

from utils.preprocess import preprocess_for_cnn

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess images

#Normalized & Reshaped
x_train = preprocess_for_cnn(x_train)
x_test = preprocess_for_cnn(x_test)


print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

def build_cnn():
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

    return model

# Print Architecture Once
model = build_cnn()
model.summary()

optimizers = {
    "adam": Adam(learning_rate = 0.0005),
    "sgd": SGD(learning_rate=0.005, momentum=0.9, nesterov=True),
    "rmsprop": RMSprop(learning_rate=0.0005)
}

results = {}

for name, opt in optimizers.items():
    print(f"\n==== Training with {name.upper()} ====")

    # Build fresh model
    model = build_cnn()

    # Compile with current optimizer
    model.compile(
    optimizer=opt,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
    # Early Stopping
    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
    start_time = time.time()

    #Train
    history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)
    
    end_time = time.time()
    training_time = end_time - start_time

    #Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"{name.upper()} Test Accuracy:", test_acc)

    results[name] = {
        "accuracy": test_acc,
        "time": training_time,
        "epochs": len(history.history["loss"])
    }

    # -------------------------------
    # Misclassfication Analysis
    # -------------------------------

    # Predict on test set
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Find misclassified indices
    misclassified = np.where(predicted_labels != y_test)[0]
    print(f"{name.upper()} Total misclassified:", len(misclassified))

    # Save Model
    model.save(f"models/cnn_{name}.keras")

    K.clear_session()

print("\n===== Final Optimizer Comparison =====")

for name, info in results.items():
    print(
        f"{name.upper()} | "
        f"Accuracy: {info['accuracy']:.4f} | "
        f"Time: {info['time']:.2f}s | "
        f"Epochs: {info['epochs']}"
    )


















