import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

## Prepare the data
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)  # Add channel dimension
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

## Function to build models with different structures
def build_model(dropout_rate=0.5):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())  # Batch Normalization for stability
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))  # Dropout for regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model

## Experiment with different parameters
# Define parameters
batch_size = 128
epochs = 10
learning_rate = 0.001
dropout_rate = 0.4

# Build and compile the model
model = build_model(dropout_rate=dropout_rate)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(lr=learning_rate),
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predict on test data
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

## Evaluation metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.show()

## Visualization
# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

## Visualize some correct and incorrect predictions
def plot_images(images, true_labels, pred_labels=None, title=None, num_images=9):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        true_label = true_labels[i]
        pred_label = pred_labels[i] if pred_labels is not None else None
        if pred_label is None:
            plt.title(f"True: {true_label}")
        else:
            plt.title(f"True: {true_label}, Pred: {pred_label}", color="green" if true_label == pred_label else "red")
        plt.axis("off")
    if title:
        plt.suptitle(title, size=16)
    plt.tight_layout()
    plt.show()

# Visualize correct predictions
correct = np.where(y_pred_classes == y_true)[0]
plot_images(x_test[correct], y_true[correct], y_pred_classes[correct], title="Correct Predictions")

# Visualize incorrect predictions
incorrect = np.where(y_pred_classes != y_true)[0]
plot_images(x_test[incorrect], y_true[incorrect], y_pred_classes[incorrect], title="Incorrect Predictions")
