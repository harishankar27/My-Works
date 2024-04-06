import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History, ModelCheckpoint, CSVLogger, EarlyStopping, Callback
from tensorflow.keras.layers import GlobalMaxPooling1D, AveragePooling1D , Input, Activation, Dense, Dropout, SpatialDropout1D, Conv1D, TimeDistributed, MaxPooling1D, Flatten, ConvLSTM2D, Bidirectional, BatchNormalization, GlobalAvgPool1D, GlobalAveragePooling1D, MaxPooling1D, LSTM, GRU
from tensorflow.keras.utils import plot_model
from livelossplot import PlotLossesKerasTF
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf

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


# Describing the model

model = Sequential([
        Input(shape=(300, 7)),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        SpatialDropout1D(0.2),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        SpatialDropout1D(0.15),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalAvgPool1D(),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n")
model.summary()

