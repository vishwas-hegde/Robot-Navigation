import observation_model as obs_model
import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import covariances
import ukf as ukf
import plots
import time

def simulate(filename: str, R = None):
    """
    Simulate the Unscented Kalman Filter.
    Input:
        filename: str: path to the .mat file containing the data
        R: np.ndarray: observation noise covariance matrix
    Output:
        estimated_data: np.ndarray: estimated data
        est_times: np.ndarray: estimated times
        gt: np.ndarray: ground truth data
        gt_tstamp: np.ndarray: ground truth timestamps
        filtered_data: np.ndarray: filtered data
        time: float: time taken to run the simulation
    """
    model = obs_model.ObsModel()
    data_list, gt, gt_tstamp = model.process_data(filename)

    positions = []
    orientations = []
    est_times = []
    data = data_list
    for idx, dat in enumerate(data):
        if isinstance(dat['id'], int):
            pos, rot = model.estimate_pose([dat['id']], idx)
            positions.append(pos)
            orientations.append(rot)
            est_times.append(float(dat['t']))
            continue
        if dat['id'].shape[0] > 1:
            pos, rot = model.estimate_pose(dat['id'], idx)
            positions.append(pos)
            orientations.append(rot)
            est_times.append(float(dat['t']))

    estimated_data = np.zeros((6, len(positions)))
    estimated_data[0] = np.array([pos[0] for pos in positions])
    estimated_data[1] = np.array([pos[1] for pos in positions])
    estimated_data[2] = np.array([pos[2] for pos in positions])
    estimated_data[3] = np.array([rot[0] for rot in orientations])
    estimated_data[4] = np.array([rot[1] for rot in orientations])
    estimated_data[5] = np.array([rot[2] for rot in orientations])

    if R is None:
        covariance = covariances.cov_compute(estimated_data, est_times, gt, gt_tstamp)

        R = covariance    # observation noise covariance matrix

    Q0 = 0.001 * np.eye(6, 6)   # process noise covariance matrix, where noise is the bias drift

    B = np.vstack((np.zeros((9, 6)), np.eye(6, 6)))

    # initial state covariance matrix
    P_pos = 0.05 ** 2 * np.eye(3)
    P_rot = 0.05 ** 2 * np.eye(3)
    P_vel = 0.05 ** 2 * np.eye(3)
    P_bias = 0.05 * np.eye(6)
    P = block_diag(P_pos, P_rot, P_vel, P_bias)   # initial state covariance matrix

    uk = ukf.UKF(P, Q0, R, 0.01)  # Unscented Kalman Filter object

    filtered_data = np.zeros((6, len(estimated_data[0])))

    t = 0   # initial time
    x = np.concatenate((gt[0:6, :][:, 0], np.zeros(9)))   # initial state
    i = 0   # index for filtered data
    start = time.time()
    for idx, dat in enumerate(data_list):
        pos = None
        rot = None
        if isinstance(dat['id'], int):
            pos, rot = model.estimate_pose([dat['id']], idx)

        elif dat['id'].shape[0] > 1:
            pos, rot = model.estimate_pose(dat['id'], idx)

        if pos is not None:
            dt = dat["t"] - t

            u = np.concatenate((dat['omg'], dat['acc']), axis=0)

            Q = (dt * B) @ Q0 @ (dt * B).T

            z = np.concatenate((pos, rot), axis=0)

            x, P = uk.predict(x, u, P, Q, dt)    # predict the next state
            x, P = uk.update(x, z, P, R)         # refine the state estimate

            t = dat["t"]

            filtered_data[:, i] = x[0:6]
            i = i + 1
    end = time.time()

    return estimated_data, est_times, gt, gt_tstamp, filtered_data, end - start


if __name__ == "__main__":
    simulate(r"D:\Spring 2024\Adv Nav\Codes\Nonlinear Kalman Filter\data\studentdata1.mat")