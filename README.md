# 3D Trajectory Calculation from IMU_Data

## 1. Project Overview

This project aims to calculate accurate 3D trajectories using data IMU sensors. The main tasks involve processing the raw data from these sensors, applying data quality checks, performing interpolations and conversions, and ultimately calculating a true movement trajectory in 3D space using advanced algorithms.

### 1.1 Description of Task and Basic Technologies

- **Overall Task**: The primary objective is to compute precise 3D trajectories from IMU data. This involves filtering, processing, and visualizing sensor data and applying robust algorithms to achieve accurate motion tracking.

- **Current Sensor Operation**:
  - **IMU Sensor**: Measures linear acceleration, angular velocity, and magnetic field in three dimensions.
  
- **Technologies to be Used**:
  - Programming Languages: Python (for data processing and algorithm implementation).
  - Libraries: NumPy, SciPy, Pandas (data handling and processing), Matplotlib, Plotly (data visualization).

## 2. IMU Data Handling

### 2.1 Data Quality Factors

- **IMU Data Quality**: Affected by sensor noise, temperature variations, mechanical vibrations, and bias.

### 2.2 Data Processing and Visualization

- **IMU Data**: Needs calibration, filtering (e.g., low-pass filter), and drift compensation.
- **Visualization**: Use 3D plots to represent data streams and trajectories.

### 2.3 Data Interpolation and Conversion

- **Interpolation**: Apply methods such as linear or spline interpolation to fill gaps in the data.

## 3. 3D Trajectory Calculation from IMU Data

### 3.1 Algorithm Exploration

- **Available Algorithms**:
  - Dead Reckoning: Simple integration of IMU data; susceptible to drift over time.
  - Kalman Filter: Optimal state estimation; requires accurate noise modeling.
  - Complementary Filter: Combines high-pass filter (accelerometer) and low-pass filter (gyroscope) data; balances noise and drift.
  - Particle Filter: Probabilistic approach; computationally intensive but robust in complex scenarios.
  
- **Advantages and Disadvantages**:
  - Dead Reckoning: Easy to implement but accumulates errors.
  - Kalman Filter: Accurate but requires careful tuning.
  - Complementary Filter: Efficient but less accurate over long periods.
  - Particle Filter: Highly accurate but requires high computational power.

## 4. Project Structure

- **src/**: This directory contains essential scripts for processing IMU data and calculating accurate 3D trajectories. Each script is dedicated to a specific aspect of the data processing pipeline, from calibration to trajectory estimation.

  - **Calibration.py**: A class-based script designed for calibrating IMU (Inertial Measurement Unit) data. It handles data loading, cleaning, and the application of advanced calibration techniques to ensure accurate sensor readings.

  - **Orientation.py**: This script focuses on accurate orientation estimation from IMU data. It encompasses data loading, cleaning, and orientation calculation using custom algorithms to provide reliable orientation metrics.

  - **Gravity.py**: A specialized script for compensating for gravity's effects during dynamic movement. It includes modules for data loading, cleaning, calibration, and precise orientation estimation, ensuring that the IMU data reflects true motion without gravitational interference.

  - **DR.py**: This script implements the Dead Reckoning (DR) algorithm to estimate the 3D trajectory in space. It integrates IMU data to provide a continuous estimation of position, although it may accumulate errors over time.

  - **EKF.py**: A sophisticated script that applies the Extended Kalman Filter (EKF) to estimate the true 3D trajectory from IMU data. It provides a more accurate trajectory estimation by optimally combining data and minimizing errors.


## 5. Getting Started

### 5.1 Prerequisites

- Python 3.8 or higher
- Libraries: NumPy, SciPy, Pandas, Matplotlib, Plotly, scikit-learn

### 5.2 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/3D-Trajectory-Calculation.git
cd 3D-Trajectory-Calculation
