'''
Developer: Sumit Gadhiya
Date: 28.07.2024
Topic: 3D Trajectory Estimate using Dead Reckoning Algorithm

This script provides an algorithm for estimating the true 3D trajectory in space from IMU data using the Dead Reckoning(DR) Algorithm method.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the data from .csv file
file_path = r"C:\Users\spptl\OneDrive\Desktop\Algoritham\q000000b6.csv"
data = pd.read_csv (file_path, delimiter= ';')
#print(data.head())


# Extract accelerometer and gyroscope data
imu_columns = [col for col in data.columns if 'accel_' in col or 'gyro_' in col or 'mag_' in col]
imu_data= data[imu_columns]
#print(imu_data.head())


#### Filtration for IMU

# Remove rows with missing values
imu_data_clean = imu_data.dropna()

# Remove duplicate rows
imu_data_clean = imu_data_clean.drop_duplicates()

# Function to replace missing or zero 25th imu_data_clean points with the average of the 23th and 24th imu_data_clean points
def replace_missing_25th_points(row):
    for axis in ['accel', 'gyro', 'mag']:
        for coord in ['x', 'y', 'z']:
            col_22 = f"{axis}_{coord}_22"
            col_23 = f"{axis}_{coord}_23"
            col_24 = f"{axis}_{coord}_24"
            if row[col_24] == 0 or pd.isna(row[col_24]):
                row[col_24] = (row[col_22] + row[col_23]) / 2
    return row

# Apply the function to the imu_data_clean frame
imu_data_clean = imu_data_clean.apply(replace_missing_25th_points, axis=1)

# Print all the 25th imu_data_clean points to verify changes
#print(imu_data_clean.head(15)[[f"{axis}_{coord}_24" for axis in ['accel', 'gyro'] for coord in ['x', 'y', 'z']]])

# Display the first few rows of the cleaned imu_data_clean
#print(imu_data_clean.head())



# Verification of total data points

# Check the number of rows
num_imu_rows = imu_data_clean.shape[0]

# Check the number of columns for each axis
num_imu_columns = len([col for col in imu_data_clean.columns if 'accel' in col or 'gyro_' in col])

# Calculate the number of values per axis
values_per_axis = num_imu_rows * 25

# Display the verification results
verification_results = {
    "Number of Rows": num_imu_rows,
    "Number Columns": num_imu_columns,
    "Values per Axis": values_per_axis,
}
#print(verification_results)


# Calculate the offsets for X, Y, and Z axes using the entire dataset
offset_x = imu_data_clean[[f"accel_x_{i}" for i in range(25)]].mean().mean()
offset_y = imu_data_clean[[f"accel_y_{i}" for i in range(25)]].mean().mean()
offset_z = imu_data_clean[[f"accel_z_{i}" for i in range(25)]].mean().mean()

#print(f"Offset X: {offset_x}, Offset Y: {offset_y}, Offset Z: {offset_z}")

# Adjust the IMU data by subtracting the offsets
for i in range(25):
    imu_data_clean[f"accel_x_{i}"] -= offset_x
    imu_data_clean[f"accel_y_{i}"] -= offset_y
    imu_data_clean[f"accel_z_{i}"] -= offset_z

# Display the first few rows of the cleaned imu_data_clean
#print(imu_data_clean.head())



# Flatten the accelerometer data for x, y, z axes
accel_x = imu_data_clean[[f"accel_x_{i}" for i in range(25)]].values.flatten()
accel_y = imu_data_clean[[f"accel_y_{i}" for i in range(25)]].values.flatten()
accel_z = imu_data_clean[[f"accel_z_{i}" for i in range(25)]].values.flatten()

# Flatten the gyroscope data for x, y, z axes
gyro_x = imu_data_clean[[f"gyro_x_{i}" for i in range(25)]].values.flatten()
gyro_y = imu_data_clean[[f"gyro_y_{i}" for i in range(25)]].values.flatten()
gyro_z = imu_data_clean[[f"gyro_z_{i}" for i in range(25)]].values.flatten()

# Flatten the magnetometer data for x, y, z axes
mag_x = imu_data_clean[[f"mag_x_{i}" for i in range(25)]].values.flatten()
mag_y = imu_data_clean[[f"mag_y_{i}" for i in range(25)]].values.flatten()
mag_z = imu_data_clean[[f"mag_z_{i}" for i in range(25)]].values.flatten()


'''
### Visulization of raw IMU data
# Plot for accelerometer imu_data_clean
plt.figure(figsize=(12, 8))

# Plot Acceleration X Axis
plt.subplot(3, 1, 1)
plt.plot(accel_x, label='Acceleration X')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

# Plot Acceleration Y Axis
plt.subplot(3, 1, 2)
plt.plot(accel_y, label='Acceleration Y', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

# Plot Acceleration Z Axis
plt.subplot(3, 1, 3)
plt.plot(accel_z, label='Acceleration Z', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot for gyroscope imu_data_clean
plt.figure(figsize=(12, 8))

# Plot Gyro X Axis
plt.subplot(3, 1, 1)
plt.plot(gyro_x, label='Gyro X')
plt.xlabel('Sample Index')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()

# Plot Gyro Y Axis
plt.subplot(3, 1, 2)
plt.plot(gyro_y, label='Gyro Y', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()

# Plot Gyro Z Axis
plt.subplot(3, 1, 3)
plt.plot(gyro_z, label='Gyro Z', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot for magnetometer imu_data_clean
plt.figure(figsize=(12, 8))

# Plot Magnetometer X Axis
plt.subplot(3, 1, 1)
plt.plot(mag_x, label='Mag X')
plt.xlabel('Sample Index')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()

# Plot Magnetometer Y Axis
plt.subplot(3, 1, 2)
plt.plot(mag_y, label='Mag Y', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()

# Plot Magnetometer Z Axis
plt.subplot(3, 1, 3)
plt.plot(mag_z, label='mag Z', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()

plt.tight_layout()
plt.show()
'''




### Dead Reckoning- Simple Extrapolation of Movement from raw IMU

# Assuming a constant sampling rate
sampling_rate = 25  # Hz
dt = 1 / sampling_rate  # Time interval

# Integrate acceleration to get velocity
velocity_x = np.cumsum(accel_x) * dt
velocity_y = np.cumsum(accel_y) * dt
velocity_z = np.cumsum(accel_z) * dt

# Integrate velocity to get position
position_x = np.cumsum(velocity_x) * dt
position_y = np.cumsum(velocity_y) * dt
position_z = np.cumsum(velocity_z) * dt

# Plot Velocity Data
plt.figure(figsize=(12, 8))

# Plot Velocity X Axis
plt.subplot(3, 1, 1)
plt.plot(velocity_x, label='Velocity X')
plt.xlabel('Sample Index')
plt.ylabel('Velocity (m/s)')
plt.legend()

# Plot Velocity Y Axis
plt.subplot(3, 1, 2)
plt.plot(velocity_y, label='Velocity Y', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Velocity (m/s)')
plt.legend()

# Plot Velocity Z Axis
plt.subplot(3, 1, 3)
plt.plot(velocity_z, label='Velocity Z', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Velocity (m/s)')
plt.legend()

plt.tight_layout()
plt.show()


# Plot the estimated positions
plt.figure(figsize=(12, 8))

# Plot Position X Axis
plt.subplot(3, 1, 1)
plt.plot(position_x, label='Position X')
plt.xlabel('Sample Index')
plt.ylabel('Position (m)')
plt.legend()

# Plot Position Y Axis
plt.subplot(3, 1, 2)
plt.plot(position_y, label='Position Y', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Position (m)')
plt.legend()

# Plot Position Z Axis
plt.subplot(3, 1, 3)
plt.plot(position_z, label='Position Z', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Position (m)')
plt.legend()

plt.tight_layout()
plt.show()

# Visualization of 3D Trajectory

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(position_x, position_y, position_z, label='3D Trajectory')
ax.set_xlabel('Position X (m)')
ax.set_ylabel('Position Y (m)')
ax.set_zlabel('Position Z (m)')
ax.legend()

plt.title('3D Trajectory using Dead reckoning')
plt.show()










