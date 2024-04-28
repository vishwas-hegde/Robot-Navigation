import numpy as np
import json
import task3
import observation_model as obs_model
from scipy.linalg import block_diag
import ukf

def noise_compare():
    """
    Compare the RMSE loss for UKF with different observation noise.
    :return: RMSE List
    """
    filename = 'data/studentdata%d.mat'
    Q0 = [0.001, 0.005, 0.01, 0.02]
    rmse_noise = {}
    R = task3.average_covarince()

    for q in Q0:
        rmse_noise[q] = {}
        for i in range(1, 8):
            file = filename % i
            rmse_est, rmse_filter = simulate(file, R, q)
            rmse_noise[q][i] = rmse_filter

    # save the data to a file
    with open('rmse_pf_noise.json', 'w') as f:
        json.dump(rmse_noise, f)

    print(json.dumps(rmse_noise, indent=4))


def simulate(filename, R, q):
    """
    Simulate the Unscented Kalman Filter.
    :param filename: path to the data file
    :return: plots the results
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


    Q0 = q * np.eye(6, 6)   # process noise covariance matrix, where noise is the bias drift

    B = np.vstack((np.zeros((9, 6)), np.eye(6, 6)))

    # initial state covariance matrix
    P_pos = 0.05 ** 2 * np.eye(3)
    P_rot = 0.05 ** 2 * np.eye(3)
    P_vel = 0.05 ** 2 * np.eye(3)
    P_bias = 0.05 * np.eye(6)
    P = block_diag(P_pos, P_rot, P_vel, P_bias)   # initial state covariance matrix

    uk = ukf.UKF(P, Q0, R, 0.01)  # Unscented Kalman Filter object

    filtered_data = np.zeros((6, len(positions)))

    t = 0   # initial time
    x = np.concatenate((gt[0:6, :][:, 0], np.zeros(9)))   # initial state
    i = 0   # index for filtered data

    for idx, dat in enumerate(data_list):
        pos = None
        rot = None
        if isinstance(dat['id'], int):
            pos, rot = model.estimate_pose([dat['id']], idx)

        elif dat['id'].shape[0] > 1:
            pos, rot = model.estimate_pose(dat['id'], idx)

        if pos is not None:
            dt = dat["t"] - t
            try:
                u = np.concatenate((dat['omg'], dat['acc']), axis=0)
            except:
                u = np.concatenate((dat['drpy'], dat['acc']), axis=0)

            Q = (dt * B) @ Q0 @ (dt * B).T

            z = np.concatenate((pos, rot), axis=0)

            x, P = uk.predict(x, u, P, Q, dt)    # predict the next state
            x, P = uk.update(x, z, P, R)         # refine the state estimate

            t = dat["t"]

            filtered_data[:, i] = x[0:6]
            i = i + 1

    rmse_est, rmse_filter = rmse_calcuate_ukf(filtered_data, est_times, gt, gt_tstamp, filtered_data)

    return rmse_est, rmse_filter


def rmse_calcuate_ukf(estimated_data, est_times, gt, gt_tstamp, filtered_data):
    """
    Compute the RMSE loss between the ground truth and the estimated data and the filtered data.
    :param estimated_data: Data estimated from camera data
    :param est_times: Times for estimated data
    :param gt: The ground truth data
    :param gt_tstamp: Times for ground truth data
    :param filtered_data: Data filtered by the Particle Filter

    :return: RMSE
    """
    rmse_est = np.zeros(len(estimated_data[0]))
    rmse_filter = np.zeros(len(filtered_data[0]))
    for i in range(len(estimated_data[0])):
        data_idx_gt = np.argmin(np.abs(gt_tstamp - est_times[i]))
        rmse_est[i] = np.sqrt(np.mean((estimated_data[:3, i] - gt[:3, data_idx_gt]) ** 2))
        rmse_filter[i] = np.sqrt(np.mean((filtered_data[:3, i] - gt[:3, data_idx_gt]) ** 2))

    return np.mean(rmse_est), np.mean(rmse_filter)


if __name__ == '__main__':
    noise_compare()