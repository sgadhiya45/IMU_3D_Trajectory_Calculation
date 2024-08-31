'''
Developer: Sumit Gadhiya
Date: 03.08.2024
Topic: 3D Trajectory Estimate using EKF Algorithm

This script provides an algorithm for estimating the true 3D trajectory in space from IMU data using the Extended Kalman Filter (EKF) method.
'''

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
raw_accel_data = flatten_sensor_data(imu_data_clean, 'accel', axes)
raw_gyro_data = flatten_sensor_data(imu_data_clean, 'gyro', axes)
raw_mag_data = flatten_sensor_data(imu_data_clean, 'mag', axes)


### Sensor calibration

# Calibration Parameters obtained from stationary measurement
accel_bias = {'x': -0.22679855803301108, 'y': -0.04908011182875657, 'z': 0.2333255275361772}
gyro_bias = {'x': 0.09507937523386988, 'y': 0.03199991259812899, 'z': 0.0313813085670967}
mag_bias = {'x': 224.20215136485703, 'y': 101.07854313115618, 'z': -21.355946192676296}

# Apply calibration (bias correction)
accel_data = {axis: raw_accel_data[axis] - accel_bias[axis] for axis in axes}
gyro_data = {axis: raw_gyro_data[axis] - gyro_bias[axis] for axis in axes}
mag_data = {axis: raw_mag_data[axis] - mag_bias[axis] for axis in axes}


#### orientation Calculation

# Low-pass filter parameters
def butter_lowpass(cutoff, fs=25, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff=0.5, fs=25, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

# Stack the data into 2D arrays for filtering
accel_data = np.column_stack((accel_data['x'], accel_data['y'], accel_data['z']))
gyro_data = np.column_stack((gyro_data['x'], gyro_data['y'], gyro_data['z']))
mag_data = np.column_stack((mag_data['x'], mag_data['y'], mag_data['z']))

# Apply low-pass filtering to the data
accel_data = apply_lowpass_filter(accel_data)
gyro_data = apply_lowpass_filter(gyro_data)
mag_data = apply_lowpass_filter(mag_data)

# Sampling rate and time step
sampling_rate = fs  = 25 # Hz
time_step = dt = 1 / 25  # timestep = dt = 1/sampling rate
# take the default value alpha = 0.98, if don't want to define Dynamic Alpha Tuning Function

# Dynamic Alpha Tuning Function
def dynamic_alpha(gyro_data, accel_data, static_threshold=0.1):
    gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
    alpha = np.ones_like(gyro_magnitude) * 0.98  # default alpha
    alpha[gyro_magnitude < static_threshold] = 0.9  # Lower alpha when nearly stationary
    return alpha

# complementary filter for sensor fusion
def complementary_filter(accel_data, gyro_data, mag_data, dt):
    roll = np.zeros_like(gyro_data[:, 0])
    pitch = np.zeros_like(gyro_data[:, 1])
    yaw = np.zeros_like(gyro_data[:, 2])
    
    alpha_values = dynamic_alpha(gyro_data, accel_data)
    
    for i in range(1, len(gyro_data)):
        # Gyro integration for roll, pitch, and yaw
        roll_gyro = roll[i-1] + gyro_data[i, 0] * dt
        pitch_gyro = pitch[i-1] + gyro_data[i, 1] * dt
        yaw_gyro = yaw[i-1] + gyro_data[i, 2] * dt
        
        # Roll and pitch estimation from accelerometer data
        roll_accel = np.arctan2(accel_data[i, 1], accel_data[i, 2])
        pitch_accel = np.arctan2(-accel_data[i, 0], np.sqrt(accel_data[i, 1]**2 + accel_data[i, 2]**2))
        
        # Yaw estimation from magnetometer data
        mag_x = mag_data[i, 0] * np.cos(pitch_accel) + mag_data[i, 2] * np.sin(pitch_accel)
        mag_y = mag_data[i, 0] * np.sin(roll_accel) * np.sin(pitch_accel) + mag_data[i, 1] * np.cos(roll_accel) - mag_data[i, 2] * np.sin(roll_accel) * np.cos(pitch_accel)
        yaw_mag = np.arctan2(mag_y, mag_x)
        
        # Dynamic complementary filter to combine gyro and accel/mag data
        alpha = alpha_values[i]
        roll[i] = alpha * roll_gyro + (1 - alpha) * roll_accel
        pitch[i] = alpha * pitch_gyro + (1 - alpha) * pitch_accel
        yaw[i] = alpha * yaw_gyro + (1 - alpha) * yaw_mag
    
    return roll, pitch, yaw

# Apply the complementary filter
roll, pitch, yaw = complementary_filter(accel_data, gyro_data, mag_data, dt)


#### Gravity compansation for sensor dynamic movement

# Function to calculate the rotation matrix from roll, pitch, yaw
def calculate_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    R = R_z @ R_y @ R_x
    return R

# Function to estimate the gravity vector in the sensor frame
def estimate_gravity_vector(roll, pitch, yaw):
    g_global = np.array([0, 0, 9.81])  # Gravity vector in global frame
    R = calculate_rotation_matrix(roll, pitch, yaw)
    g_sensor = R @ g_global
    return g_sensor

# Function to remove gravity from accelerometer data
def remove_gravity(accel_data, roll, pitch, yaw):
    accel_motion = np.zeros_like(accel_data)
    for t in range(len(accel_data)):
        g_sensor = estimate_gravity_vector(roll[t], pitch[t], yaw[t])
        accel_motion[t, :] = accel_data[t, :] - g_sensor
    return accel_motion

# Low-pass filter parameters 
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff=0.5, fs=25, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

# Removing gravity from the accelerometer data using the complementary filter results
linear_accel = remove_gravity(accel_data, roll, pitch, yaw)

# Apply low-pass filtering to smooth the data
filtered_linear_accel = apply_lowpass_filter(linear_accel)


# Extended Kalman Filter for 3D Trajectory Estimation

class ExtendedKalmanFilter:
    def __init__(self, dt, state_dim, measurement_dim, process_noise_cov, measurement_noise_cov, initial_state_cov):
        self.dt = dt
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Initialize state estimate and covariance
        self.state_estimate = np.zeros((state_dim, 1))
        self.state_covariance = np.eye(state_dim) * initial_state_cov
        
        # Process and measurement noise covariance matrices
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov

    def state_transition(self, state):
        # Define the state transition function
        # x_k = A * x_(k-1) + B * u_(k-1) + w_k
        A = np.eye(self.state_dim)
        A[:3, 3:] = np.eye(3) * self.dt  # Assuming constant velocity model
        
        return A @ state

    def jacobian_state_transition(self, state):
        # Jacobian of the state transition function with respect to state
        A = np.eye(self.state_dim)
        A[:3, 3:] = np.eye(3) * self.dt
        return A

    def observation_model(self, state):
        # Observation model (what we expect to observe)
        H = np.eye(self.measurement_dim, self.state_dim)
        return H @ state

    def jacobian_observation_model(self, state):
        # Jacobian of the observation model
        return np.eye(self.measurement_dim, self.state_dim)

    def predict(self):
        # Predict the next state
        F = self.jacobian_state_transition(self.state_estimate)
        self.state_estimate = self.state_transition(self.state_estimate)
        self.state_covariance = F @ self.state_covariance @ F.T + self.process_noise_cov

    def update(self, measurement):
        # Update the state estimate based on the measurement
        H = self.jacobian_observation_model(self.state_estimate)
        innovation = measurement - self.observation_model(self.state_estimate)
        innovation_covariance = H @ self.state_covariance @ H.T + self.measurement_noise_cov
        kalman_gain = self.state_covariance @ H.T @ inv(innovation_covariance)
        self.state_estimate = self.state_estimate + kalman_gain @ innovation
        self.state_covariance = (np.eye(self.state_dim) - kalman_gain @ H) @ self.state_covariance

# Define parameters
dt = 0.01  # Time step
state_dim = 6  # [x, y, z, vx, vy, vz]
measurement_dim = 3  # [x, y, z]

# Process noise and measurement noise covariances
process_noise_cov = np.eye(state_dim) * 1e-4
measurement_noise_cov = np.eye(measurement_dim) * 1e-2
initial_state_cov = 1e-2

# Initialize the EKF
ekf = ExtendedKalmanFilter(dt, state_dim, measurement_dim, process_noise_cov, measurement_noise_cov, initial_state_cov)

# Prepare to store the estimated trajectory
estimated_trajectory = []

# Loop through IMU data
for i in range(len(imu_data_clean)):
    # Assume the measurements are the position from accelerometer (after double integration) for simplicity
    measurement = imu_data_clean.iloc[i, :3].values.reshape(-1, 1)
    
    ekf.predict()
    ekf.update(measurement)
    
    # Store the estimated position (x, y, z)
    estimated_trajectory.append(ekf.state_estimate[:3].flatten())

estimated_trajectory = np.array(estimated_trajectory)

# 3D Trajectory Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2], label='Estimated Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectory Estimation using EKF')
plt.legend()
plt.show()
