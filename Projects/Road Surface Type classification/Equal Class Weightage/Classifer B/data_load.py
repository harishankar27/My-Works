import numpy as np
import pandas as pd
import os

from tqdm import tqdm
import matplotlib.pyplot as plt


n_data_sets = 9
directory = []

# We will be looking the vehicular sensor data obtained from the the linear and angular acceleartion data from the acceleration

# We will be using the accelerometer and gyro data from the below_suspension
data_left  = {}
data_right = {}
data_labels = {}

columns_to_read = ["acc_x_below_suspension", "acc_y_below_suspension", "acc_z_below_suspension","gyro_x_below_suspension", "gyro_y_below_suspension","gyro_z_below_suspension","speed"]

for i in range(1,n_data_sets + 1):
    directory.append(rf"G:\IIT H\ACADEMIC\SEM-5\Machine Learning\Project\DATA\PVS {i}")

iterable = range(0,n_data_sets)
for i in tqdm(iterable, desc="Taking in Input", unit="data_sets"):
    dir = os.path.join(directory[i],"dataset_gps_mpu_left.csv")
    df = pd.read_csv(dir , usecols=columns_to_read)
    data_left.update({f"PVS_{i+1}": df})

print("Taking in data from left sensors:completed\n ")

for i in tqdm(iterable, desc="Taking in Input", unit="data_sets"):
    dir = os.path.join(directory[i],"dataset_gps_mpu_right.csv")
    df = pd.read_csv(dir , usecols=columns_to_read)
    data_right.update({f"PVS_{i+1}": df})

#We make slight changes in the labeling: -1 = dirt road, 0=cobblestone_road, 1:asphalt_road
print("Taking in data from right sensors:completed\n")

for i in tqdm(iterable, desc="Taking in Input", unit="data_sets"):
    dir = os.path.join(directory[i], "dataset_labels.csv")
    df = pd.read_csv(dir, usecols=["dirt_road", "cobblestone_road", "asphalt_road"])


    data_labels.update({f"PVS_{i+1}": df})
print("Taking in data labels:completed")
    #print(data_labels[f"PVS_{i+1}"])

