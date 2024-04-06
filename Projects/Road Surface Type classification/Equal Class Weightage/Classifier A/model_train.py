from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from model import *
from preprocess import *

# Clear the session and close any open TensorFlow resources
K.clear_session()


model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Early stopping to prevent overfitting and stop training when the validation loss stops improving
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train the model with callbacks
Model = model.fit(X_train, y_train[:, :], epochs=200, validation_data=(X_val, y_val[:, :]),
                    callbacks=[model_checkpoint])

# Access the training and validation loss from the history
train_loss = Model.history['loss']
val_loss = Model.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plot the training and validation loss
# plt.scatter(epochs, train_loss, label='Training Loss')
# plt.scatter(epochs, val_loss, label='Validation Loss')
plt.plot(epochs, train_loss,label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss for Classifier A')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training and validation loss.png")
plt.show()
