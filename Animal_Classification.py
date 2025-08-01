import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import os


print("Loading and preprocessing the dataset...")

# Define paths and parameters
dataset_dir = 'Animals'
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Use tf.keras.utils.image_dataset_from_directory for loading data

full_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Get class names
class_names = full_dataset.class_names
print("Class Names:", class_names)

# Split the dataset into training, validation, and test sets
dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.8 * dataset_size) # 80% for training and validation initially
test_size = int(0.1 * dataset_size) # 10% for testing
validation_size = dataset_size - train_size - test_size 

train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size).take(test_size)
validation_dataset = full_dataset.skip(train_size + test_size)


# Apply rescaling to the datasets
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess)
validation_dataset = validation_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)


print("\nDesigning the 2-layer CNN model...")


model = Sequential([
    # First Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    # Second Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    Flatten(),

    # Dense Layer
    Dense(512, activation='relu'),
    Dropout(0.5), # Dropout for regularization

    # Output Layer
    Dense(len(class_names), activation='softmax') 
])

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



print("\nTraining the model...")


# Set a smaller number of epochs for a quick demonstration
EPOCHS = 15

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)


print("\nEvaluating the model on the test set...")
test_loss, test_acc = model.evaluate(test_dataset)
print(f'\nTest accuracy: {test_acc:.2f}')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



print("\nVisualizing predictions...")


# Get a batch of test images

for x_test, y_test in test_dataset.take(1):
    predictions = model.predict(x_test)

    plt.figure(figsize=(15, 15))
    for i in range(min(9, BATCH_SIZE)): 
        plt.subplot(3, 3, i + 1)
        # Convert image back to uint8 for visualization
        plt.imshow(tf.cast(x_test[i] * 255.0, tf.uint8))
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[np.argmax(y_test[i])]

        # Set title color based on correctness
        title_color = 'g' if predicted_class == true_class else 'r'

        plt.title(f'True: {true_class}\nPred: {predicted_class}', color=title_color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Save the model file as required by the deliverables
model.save('animal_classifier_model.h5')
print("\nModel saved as animal_classifier_model.h5")