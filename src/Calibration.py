'''
Developer: Sumit Gadhiya
Date: 19.06.2024
Topic: Sensor calibration

This script provides a class-based approach to calibrating IMU (Inertial Measurement Unit) data,
including data loading, cleaning, and the application of custom calibration techniques.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the stationary dataset
file_path = r"C:\Users\spptl\OneDrive\Desktop\Algoritham\q000000b6.csv"
data = pd.read_csv(file_path, delimiter=';')

# Extract accelerometer, gyroscope, and magnetometer data
imu_columns = [col for col in data.columns if 'accel_' in col or 'gyro_' in col or 'mag_' in col]
imu_data = data[imu_columns]


#### Basic Filtration for IMU

# Remove rows with missing values
imu_data_clean = imu_data.dropna()

# Remove duplicate rows
imu_data_clean = imu_data_clean.drop_duplicates()

# Function to replace missing or zero 25th points with the average of the 23rd and 24th points
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

# Function to flatten sensor data for given axes and sensor type
def flatten_sensor_data(data, sensor_type, axes, points=25):
    return {axis: data[[f"{sensor_type}_{axis}_{i}" for i in range(points)]].values.flatten() for axis in axes}

# Define axes
axes = ['x', 'y', 'z']

# Flatten data for accelerometer, gyroscope, and magnetometer
accel_data = flatten_sensor_data(imu_data_clean, 'accel', axes)
gyro_data = flatten_sensor_data(imu_data_clean, 'gyro', axes)
mag_data = flatten_sensor_data(imu_data_clean, 'mag', axes)

#### Bias calculation

# Function to calculate biases for each axis of a sensor
def calculate_biases(sensor_data, axes):
    return {axis: float(np.mean(sensor_data[axis])) for axis in axes}

# Calculate biases for each sensor type
accel_bias = calculate_biases(accel_data, axes)
gyro_bias = calculate_biases(gyro_data, axes)
mag_bias = calculate_biases(mag_data, axes)

# Print the results
print("Accelerometer Bias:", accel_bias)
print("Gyroscope Bias:", gyro_bias)
print("Magnetometer Bias:", mag_bias)

# Function to apply bias correction to sensor data
def apply_bias_correction(sensor_data, sensor_bias, axes, gravity_on_z=False):
    calibrated_data = {}
    for axis in axes:
        if axis == 'z' and gravity_on_z:
            calibrated_data[axis] = sensor_data[axis] - sensor_bias[axis] + 9.81
        else:
            calibrated_data[axis] = sensor_data[axis] - sensor_bias[axis]
    return calibrated_data

# Apply bias correction to the data, keeping gravity on the Z-axis of the accelerometer
accel_calibrated = apply_bias_correction(accel_data, accel_bias, axes, gravity_on_z=True)
gyro_calibrated = apply_bias_correction(gyro_data, gyro_bias, axes)
mag_calibrated = apply_bias_correction(mag_data, mag_bias, axes)


# Validation of calibration
# Function to plot the data before and after calibration
def plot_comparison(sensor_data_before, sensor_data_after, sensor_type, axes):
    fig, axs = plt.subplots(len(axes), 1, figsize=(10, 12))
    fig.suptitle(f'{sensor_type} Data Before and After Calibration')

    for i, axis in enumerate(axes):
        axs[i].plot(sensor_data_before[axis], label='Before Calibration', alpha=0.6)
        axs[i].plot(sensor_data_after[axis], label='After Calibration', alpha=0.6)
        axs[i].set_title(f'{sensor_type} - {axis.upper()} Axis')
        axs[i].legend(loc='best', fontsize='large')

    plt.tight_layout()
    plt.show()

# Prepare data for plotting
accel_before = {'x': accel_data['x'], 'y': accel_data['y'], 'z': accel_data['z']}
accel_after = {'x': accel_calibrated['x'], 'y': accel_calibrated['y'], 'z': accel_calibrated['z']}

gyro_before = {'x': gyro_data['x'], 'y': gyro_data['y'], 'z': gyro_data['z']}
gyro_after = {'x': gyro_calibrated['x'], 'y': gyro_calibrated['y'], 'z': gyro_calibrated['z']}

mag_before = {'x': mag_data['x'], 'y': mag_data['y'], 'z': mag_data['z']}
mag_after = {'x': mag_calibrated['x'], 'y': mag_calibrated['y'], 'z': mag_calibrated['z']}

# Plot the comparisons for each sensor
plot_comparison(accel_before, accel_after, 'Accelerometer', axes)
plot_comparison(gyro_before, gyro_after, 'Gyroscope', axes)
plot_comparison(mag_before, mag_after, 'Magnetometer', axes)
