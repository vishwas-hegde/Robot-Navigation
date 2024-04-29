import observation_model as obs_model
import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import ukf as ukf
from haversine import haversine
import matplotlib.pyplot as plt

def simulate(filename, system, R = None):
    """
    Simulate the Unscented Kalman Filter.
    :param filename: path to the data file
    :param system: error_state or full_state
    :param R: measurement covariance matrix
    :return: plots the results
    """
    model = obs_model.Observation(filename)
    data = model.read_data()

    time = data[:, 0]
    true_pos = data[:, 1:4]
    true_rot = data[:, 4:7]
    imu_w = data[:, 7:10]
    imu_f = data[:, 10:13]
    gps_pos = data[:, 13:16]
    gps_vel = data[:, 16:19]

    prev_pos = gps_pos[0]
    prev_rot = true_rot[0]
    prev_vel = gps_vel[0]

    # R = covariance    # observation noise covariance matrix
    R = np.eye(6,6) * 0.1

    Q0 = 20 * np.eye(6, 6)   # process noise covariance matrix, where noise is the bias drift

    # initial state covariance matrix
    P_pos = 5 ** 2 * np.eye(3)
    P_rot = 5 ** 2 * np.eye(3)
    P_vel = 5 ** 2 * np.eye(3)

    if system == "error_state":
        P_bias = 5 * np.eye(3)
        B = np.vstack((np.zeros((9, 6)), np.eye(3, 6)))
        x = np.concatenate((prev_pos, prev_rot, prev_vel, np.ones(3)))  # initial state
    elif system == "full_state":
        P_bias = 5 * np.eye(6)
        B = np.vstack((np.zeros((9, 6)), np.eye(6, 6)))
        x = np.concatenate((prev_pos, prev_rot, prev_vel, np.ones(6)*1))  # initial state

    P = block_diag(P_pos, P_rot, P_vel, P_bias)   # initial state covariance matrix

    uk = ukf.UKF(P, Q0, R, 1)  # Unscented Kalman Filter object

    filtered_data = np.zeros((3, len(time)))

    for i in range(len(time)):
        dt = 1

        u = np.concatenate((imu_w[i], imu_f[i]), axis=0)    # control input

        Q = (dt * B) @ Q0 @ (dt * B).T

        z = np.concatenate((gps_pos[i], gps_vel[i]), axis=0)     # measurement

        x, P = uk.predict(model, x, u, P, Q, dt, gps_pos[i], system)    # predict the next state
        x, P = uk.update(x, z, P, R, system)         # refine the state estimate

        filtered_data[:, i] = x[0:3]

    haversine_error(filtered_data, true_pos)
    # plots(filtered_data.T, true_pos)

def haversine_error(filtered_data, true_data):
    """
    Calculate the haversine error.
    :return: haversine error
    """
    # calculate the error
    filtered_data = filtered_data.T
    # calculate the haversine error
    have = []
    for i in range(len(filtered_data)):
        have.append(haversine((filtered_data[i][0], filtered_data[i][1]), (true_data[i][0], true_data[i][1])))
    print("Mean Haversine Distance: ", np.mean(have))

    # Plot the haversine error vs time
    plt.figure(figsize=(14, 5))
    plt.plot(have)
    plt.xlabel("Time")
    plt.ylabel("Haversine Distance")
    plt.title("Haversine Distance vs Time")
    plt.show()

    return np.mean(have)

def plots(filtered_data, true_data):
    """
    Plot the data.
    :return:
    """
    plt.figure()
    plt.plot(filtered_data[:, 0], filtered_data[:, 1], label="Filtered Data")
    plt.plot(true_data[:, 0], true_data[:, 1], label="True Data")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Trajectory")
    plt.legend()
    plt.show()

    # Compare each state in subfigures
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(filtered_data[:, 0], label="Filtered Data")
    plt.plot(true_data[:, 0], label="True Data")
    plt.xlabel("Time")
    plt.ylabel("Latitude")
    plt.title("Latitude")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(filtered_data[:, 1], label="Filtered Data")
    plt.plot(true_data[:, 1], label="True Data")
    plt.xlabel("Time")
    plt.ylabel("Longitude")
    plt.title("Longitude")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(filtered_data[:, 2], label="Filtered Data")
    plt.plot(true_data[:, 2], label="True Data")
    plt.xlabel("Time")
    plt.ylabel("Altitude")
    plt.title("Altitude")
    plt.legend()
    plt.show()

    # Plot the error in each state
    plt.figure(figsize=(14, 5))
    plt.plot(filtered_data[:, 0] - true_data[:, 0], label="Latitude Error")
    plt.plot(filtered_data[:, 1] - true_data[:, 1], label="Longitude Error")
    plt.plot(filtered_data[:, 2] - true_data[:, 2], label="Altitude Error")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Error in each state")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    simulate(r"trajectory_data.csv","full_state")