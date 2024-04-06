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

# # Define your model

model = models.Sequential([
    layers.InputLayer(input_shape=(16,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Display the model summary
model.summary()

# Train the model using your data
