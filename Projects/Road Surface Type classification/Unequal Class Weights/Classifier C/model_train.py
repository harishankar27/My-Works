from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from model import *
from preprocess import *

# Clear the session and close any open TensorFlow resources
K.clear_session()


model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

class_labels = np.argmax(y_train, axis=1)

# Calculate unique classes and their counts
classes, counts = np.unique(class_labels, return_counts=True)
# Step 2: Calculate Inverse Class Frequencies
inverse_frequencies = 1 / counts

# Step 3: Normalize Weights
class_weights = inverse_frequencies / np.sum(inverse_frequencies)

class_weights_dict = dict(zip(classes, class_weights))

Model = model.fit(X_train, y_train[:, :], epochs=200, validation_data=(X_val, y_val[:, :]),
                  callbacks=[model_checkpoint], class_weight=class_weights_dict)
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
plt.legend()
plt.savefig("training and validation loss_C.png")
plt.show()
