from data_load import *
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
import numpy as np


window_size = 300  #  window size as needed
n_features = 7
scaler =  MinMaxScaler(feature_range=(-1, 1))


train_set = [1,2,4,5,7,8]
n_data_points_train = 0

z=0
for i in train_set:
    n = ((data_left[f"PVS_{i}"]).shape)[0]
    z = z+n
    n_data_points_train = n_data_points_train + (n // window_size)

train_data = np.zeros((n_data_points_train,window_size,n_features))
print(z)

n_window = 0
n_x = 0
for i in train_set:
    df = data_left[f"PVS_{i}"]
    n = (((data_left[f"PVS_{i}"]).shape)[0] )//window_size
    # n_x = n_x +n
    for j in range(0, n):
        window = df.iloc[j*window_size:(j+1)*window_size]
        normalized_window = pd.DataFrame(scaler.fit_transform(window), columns=window.columns)
        train_data[n_window,:,:] = normalized_window.values
        #train_data[n_window,:,:] = window.values
        n_window = n_window + 1

n_labels_train = np.zeros((n_data_points_train,3),dtype=int)

n_window = 0
for i in train_set:
    df = data_labels[f"PVS_{i}"]
    n = (((data_labels[f"PVS_{i}"]).shape)[0] )//window_size
    for j in range(0,n):
        n_labels_train[n_window,:] = df.values[(j+1)*window_size -1,:] 
        n_window = n_window +1

# for i in range(0,n_data_points_test):
#     print(n_labels_test[i])
#     print("\n")
print("Training data is processeed \n")

# Now we split the data into training and validation sets




# Shuffle and split the data
X_train, X_val, y_train, y_val = train_test_split(train_data, n_labels_train, test_size=0.2, shuffle=True, random_state=42)


# Now X_train, y_train, X_val, and y_val are shuffled and can be used for training and validation


# Now we declare the testing data

test_set = [3,6,9]
n_data_points_test = 0

for i in test_set:
    n = ((data_left[f"PVS_{i}"]).shape)[0]
    n_data_points_test = n_data_points_test + (n // window_size)

test_data = np.zeros((n_data_points_test,window_size,n_features))


n_window = 0
n_x = 0
for i in test_set:
    df = data_left[f"PVS_{i}"]
    n = (((data_left[f"PVS_{i}"]).shape)[0] )//window_size
    # n_x = n_x +n
    for j in range(0, n):
        window = df.iloc[j*window_size:(j+1)*window_size]
        normalized_window = pd.DataFrame(scaler.fit_transform(window), columns=window.columns)
        test_data[n_window,:,:] = normalized_window.values
        # test_data[n_window,:,:] = window.values
        n_window = n_window + 1

n_labels_test = np.zeros((n_data_points_test,3), dtype=int)

n_window = 0
for i in test_set:
    df = data_labels[f"PVS_{i}"]
    n = (((data_labels[f"PVS_{i}"]).shape)[0] )//window_size
    for j in range(0,n):
        n_labels_test[n_window,:] = df.values[(j+1)*window_size -1,:] 
        n_window = n_window +1

