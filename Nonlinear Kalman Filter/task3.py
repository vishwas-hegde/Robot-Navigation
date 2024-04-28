import numpy as np
import matplotlib.pyplot as plt
import os
import observation_model as obs_model

def cov_compute(est_data, est_times, gt_data, gt_tstamp):
    """
    Compute the covariance matrix
    Inputs:
        est_data: estimated data
        est_times: times for estimated data
        gt_data: ground truth data
        gt_tstamp: times for ground truth data
    Returns:
        average_cov: the average covariance matrix
    """
    cov = []
    for idx, dat in enumerate(est_data.T):
        # find the corresponding ground truth data
        data_idx_gt = np.argmin(np.abs(gt_tstamp - est_times[idx]))
        gt_datum = gt_data[:, data_idx_gt]
        gt_datum = np.array([gt_datum[0], gt_datum[1], gt_datum[2], gt_datum[3], gt_datum[4], gt_datum[5]])

        error = gt_datum.reshape(6, 1) - dat.reshape(6, 1)  # error
        covariance = error @ error.T
        cov.append(covariance)

    average_cov = (1/(len(cov)-1)) * np.sum(cov, axis=0)   # average covariance matrix

    return average_cov

def estimate_covariances(filename):
    """
    Estimate the covariance matrix
    Inputs:
        filename: the filename of the data
    Returns:
        R: the covariance matrix
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
    R = cov_compute(estimated_data, est_times, gt, gt_tstamp)

    return R

def average_covarince():
    """
    Compute the average covariance matrix
    Returns:
        R_avg: the average covariance matrix
    """
    filename = 'data/studentdata%d.mat'

    R = []
    for i in range(1, 8):
        file = filename % i
        R.append(estimate_covariances(file))

    R_avg = np.mean(R, axis=0)

    return R_avg








