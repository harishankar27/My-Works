from data_load import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np



def data_correlation(c_matrix,size):

    data = np.zeros(15)
    row = 0
    column_start = 1
    count = 0
    while (row<size):
        column = column_start
        while(column<size):
            data[count] = c_matrix[row,column]
            column = column + 1
            count = count + 1
        column_start = column_start + 1
        row = row +1
    return data
    
    


window_size = 300  #  window size as needed
n_features = 7
scaler = RobustScaler()  #MinMaxScaler(feature_range=(-1, 1))

train_set = [1,2,4,5,7,8]
n_data_points_train = 0

z=0
for i in train_set:
    n = ((data_left[f"PVS_{i}"]).shape)[0]
    z = z+n
    n_data_points_train = n_data_points_train + (n // window_size)
    

data_matrix_size = 49

# train_data = np.zeros((n_data_points_train,21,1))
train_data = np.zeros((n_data_points_train,16,1))


n_window = 0
n_x = 0
for i in train_set:
    df = data_left[f"PVS_{i}"]
    n = (((data_left[f"PVS_{i}"]).shape)[0] )//window_size
    # n_x = n_x +n
    for j in range(0, n):
        window = df.iloc[j*window_size:(j+1)*window_size]
        normalized_window = pd.DataFrame(scaler.fit_transform(window), columns=window.columns)
        matrix = normalized_window.corr()
        matrix_correlation = data_correlation(matrix.values[0:6,0:6],6)
        train_data[n_window,0:15,0] = matrix_correlation
        train_data[n_window,15,0] =  window['speed'].mean()
        
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


# Sample data
num_instances = n_data_points_train
sequence_length = window_size
feature_dim = 7


# Shuffle and split the data
X_train, X_val, y_train, y_val = train_test_split(train_data, n_labels_train, test_size=0.2, shuffle=True, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

# Now X_train, y_train, X_val, and y_val are shuffled and can be used for training and validation


# Now we declare the testing data



window_size = 300  #  window size as needed
n_features = 7
scaler =  MinMaxScaler(feature_range=(-1, 1))

test_set = [3,6,9]
n_data_points_test = 0

for i in test_set:
    n = ((data_left[f"PVS_{i}"]).shape)[0]
    n_data_points_test = n_data_points_test + (n // window_size)

test_data = np.zeros((n_data_points_test,16,1))

correlation_matrix = np.zeros(15)
n_window = 0
n_x = 0
for i in test_set:
    df = data_left[f"PVS_{i}"]
    n = (((data_left[f"PVS_{i}"]).shape)[0] )//window_size
    # n_x = n_x +n
    for j in range(0, n):
        window = df.iloc[j*window_size:(j+1)*window_size]
        normalized_window = pd.DataFrame(scaler.fit_transform(window), columns=window.columns)
        matrix = normalized_window.corr()
        matrix_correlation = data_correlation(matrix.values[0:6,0:6],6)
        test_data[n_window,0:15,0] = matrix_correlation
        test_data[n_window,15,0] =  window['speed'].mean()
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

# for i in range(0,n_data_points_test):
#     print(n_labels_test[i])
#     print("\n")

