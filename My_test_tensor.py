import tensorflow as tf
import matplotlib.pyplot as plt

# Define data directories
train_dir = 'C:\\TensorflowImages\\train'
test_dir = 'C:\\TensorflowImages\\test'

img_height = 224
img_width = 224
batch_size = 32
num_classes = 4  # Update the number of classes

# Preprocess and load data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Define your model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile your model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define a callback to display images during training
class ImageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        plt.figure(figsize=(10, 10))
        for images, labels in train_data.take(1):
            print("Number of images in this batch:", len(images))
            for i in range(min(9, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                print("Shape of image:", images[i].shape)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title('Class: {}'.format(tf.argmax(labels[i]).numpy()))
                plt.axis("off")
        plt.show()

# Train your model with the ImageCallback
history = model.fit(train_data, epochs=10, validation_data=test_data, callbacks=[ImageCallback()])

# Visualize the training process
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
