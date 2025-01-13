import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
if not os.path.exists('./logs'):
    os.makedirs('./logs')


# Hyperparameters
img_h = img_w = 28  # MNIST images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
n_classes = 10  # Number of classes, one class per digit
n_channels = 1
epochs = 10  # Total number of training epochs
batch_size = 100  # Training batch size
lr = 0.001  # The optimization initial learning rate
display_freq = 100  # Frequency of displaying the training results
logs_path = "./logs"  # path to the folder that we want to save the logs for Tensorboard

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = np.expand_dims(x_train, -1).astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, -1).astype(np.float32) / 255.0

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

# Network Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=5, strides=1, padding="same", activation="relu",
                           input_shape=(img_h, img_w, n_channels)),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"),

    tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(n_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)

# Train the model
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    verbose=1, callbacks=[tensorboard_callback])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test loss: {test_loss:.2f}, test accuracy: {test_acc:.2%}")

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Test predictions and plot some results
y_pred = model.predict(x_test)


def plot_images(images, cls_true, cls_pred=None, title=None):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(images[i]), cmap='binary')
        if cls_pred is None:
            ax_title = f"True: {cls_true[i]}"
        else:
            ax_title = f"True: {cls_true[i]}, Pred: {cls_pred[i]}"
        ax.set_title(ax_title)
        ax.set_xticks([]), ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


# Plot correct predictions
cls_true = np.argmax(y_test, axis=1)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(x_test, cls_true, cls_pred, title='Correct Predictions')

# Plot misclassified examples
incorrect = np.logical_not(np.equal(cls_pred, cls_true))
incorrect_images = x_test[incorrect]
cls_pred_incorrect = cls_pred[incorrect]
cls_true_incorrect = cls_true[incorrect]

plot_images(incorrect_images[:9], cls_true_incorrect[:9], cls_pred_incorrect[:9], title='Misclassified Examples')
