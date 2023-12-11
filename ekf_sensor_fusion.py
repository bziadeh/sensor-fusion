import numpy as np
import matplotlib.pyplot as plt

# Simulated sensor measurements (GPS and IMU)
gps_measurement = np.array([10, 20])  # [x, y] coordinates
imu_measurement = np.array([0.1, 0.2])  # [velocity_x, velocity_y]

# Define the state vector [x, y, velocity_x, velocity_y]
state_vector = np.array([0, 0, 0, 0], dtype=float)

# Define the state transition matrix A
A = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)

# Define the observation matrix H for GPS
H_gps = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=float)

# Define the observation matrix H for IMU
H_imu = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)

# Define the process noise covariance Q
Q = np.eye(4) * 0.1

# Define the measurement noise covariance for GPS R_gps
R_gps = np.eye(2) * 1

# Define the measurement noise covariance for IMU R_imu
R_imu = np.eye(2) * 0.01

# Initialize the covariance matrix P
P = np.eye(4)

# Lists to store results for plotting
estimated_states = [state_vector[:2]]
gps_measurements = [gps_measurement]
imu_measurements = [imu_measurement]

# Number of time steps
num_steps = 100

# Simulation loop
for _ in range(num_steps):
    # Simulate sensor measurements (varying with time for demonstration)
    gps_measurement += np.random.multivariate_normal([0, 0], R_gps)
    imu_measurement += np.random.multivariate_normal([0, 0], R_imu)

    # Prediction step
    state_vector = np.dot(A, state_vector)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Update step for GPS
    K_gps = np.dot(np.dot(P, H_gps.T), np.linalg.inv(np.dot(np.dot(H_gps, P), H_gps.T) + R_gps))
    state_vector = state_vector + np.dot(K_gps, gps_measurement - np.dot(H_gps, state_vector))
    P = np.dot((np.eye(4) - np.dot(K_gps, H_gps)), P)

    # Update step for IMU
    K_imu = np.dot(np.dot(P, H_imu.T), np.linalg.inv(np.dot(np.dot(H_imu, P), H_imu.T) + R_imu))
    state_vector = state_vector + np.dot(K_imu, imu_measurement - np.dot(H_imu, state_vector))
    P = np.dot((np.eye(4) - np.dot(K_imu, H_imu)), P)

    # Store results for plotting
    estimated_states.append(state_vector[:2])
    gps_measurements.append(gps_measurement)
    imu_measurements.append(imu_measurement)

# Convert lists to numpy arrays for plotting
estimated_states = np.array(estimated_states)
gps_measurements = np.array(gps_measurements)
imu_measurements = np.array(imu_measurements)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(estimated_states[:, 0], estimated_states[:, 1], label='Estimated Position', marker='o')
plt.plot(gps_measurements[:, 0], gps_measurements[:, 1], label='GPS Measurements', marker='x')
plt.plot(imu_measurements[:, 0], imu_measurements[:, 1], label='IMU Measurements', marker='s')
plt.title('Sensor Fusion using Extended Kalman Filter')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
