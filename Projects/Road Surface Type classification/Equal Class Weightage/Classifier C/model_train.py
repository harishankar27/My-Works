from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from model import *
from preprocess import *


# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Get the list of physical devices, including GPUs
physical_devices = tf.config.list_physical_devices()
print("Physical devices:", physical_devices)

# Check if GPU is available
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU available:", gpu_devices)
    for device in gpu_devices:
        print("Device name:", device.name)
else:
    print("No GPU available.")






# Clear the session and close any open TensorFlow resources
K.clear_session()


model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Early stopping to prevent overfitting and stop training when the validation loss stops improving
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)\
# Train the model with callbacks
Model = model.fit(X_train, y_train[:, :], epochs=200, validation_data=(X_val, y_val[:, :]),
                    callbacks=[model_checkpoint])

# Access the training and validation loss from the history
train_loss = Model.history['loss']
val_loss = Model.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plot the training and validation loss
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')

plt.title('Training and Validation Loss for Classifier C')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("training and validation loss_C")
plt.legend()
plt.show()
