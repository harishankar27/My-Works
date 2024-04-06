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
from tensorflow.keras import layers, models
import tensorflow as tf


from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout



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
