import matplotlib.pyplot as plt
import numpy as np
import observation_model as obs_model

def plot_results(est_data, est_time, gt_data, gt_time, filter_data):
    """
    Plot comparison between the estimated data, the ground truth data and the filtered data.
    :param est_data: Data estimated from camera data
    :param est_time: Times for estimated data
    :param gt_data: The ground truth data
    :param gt_time: Times for ground truth data
    :param filter_data: Data filtered by the Kalman Filter
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.plot(gt_data[0], gt_data[1], gt_data[2], c='g', label='Ground Truth')
    ax.plot(est_data[0], est_data[1], est_data[2], 'r--', label='Estimated')
    ax.plot(filter_data[0], filter_data[1], filter_data[2], label='Filtered')
    ax.legend()
    ax.set_title('3D Plot for Position')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot to compare estimated position with ground truth
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(gt_time, gt_data[0])
    ax[0].plot(est_time, est_data[0])
    ax[0].plot(est_time, filter_data[0])
    ax[0].set_ylabel('X')
    # ax[0].legend()

    ax[1].plot(gt_time, gt_data[1])
    ax[1].plot(est_time, est_data[1])
    ax[1].plot(est_time, filter_data[1])
    ax[1].set_ylabel('Y')
    # ax[1].legend()

    ax[2].plot(gt_time, gt_data[2], label='Ground Truth')
    ax[2].plot(est_time, est_data[2], label='Estimated')
    ax[2].plot(est_time, filter_data[2], label='Filtered')
    ax[2].set_ylabel('Z')
    ax[2].set_xlabel('Time')
    ax[2].legend()
    ax[0].set_title('Position vs Time')


    # Plot to compare estimated angle with ground truth
    fig, ax = plt.subplots(3, 1)
    # ax[0].set_ylim(-np.pi / 2, np.pi / 2)
    ax[0].plot(gt_time, gt_data[3])
    ax[0].plot(est_time, est_data[5])
    ax[0].plot(est_time, filter_data[3])
    ax[0].set_ylabel('Roll')
    # ax[0].legend()

    # ax[1].set_ylim(-np.pi / 2, np.pi / 2)
    ax[1].plot(gt_time, gt_data[4])
    ax[1].plot(est_time, est_data[4])
    ax[1].plot(est_time, filter_data[4])
    ax[1].set_ylabel('Pitch')
    # ax[1].legend()

    # ax[2].set_ylim(-np.pi / 2, np.pi / 2)
    ax[2].plot(gt_time, gt_data[5], label='Ground Truth')
    ax[2].plot(est_time, est_data[3], label='Estimated')
    ax[2].plot(est_time, filter_data[5], label='Filtered')
    ax[2].set_ylabel('Yaw')
    ax[2].legend()
    ax[0].set_title('Orientation vs Time')

    a = rmse_loss(est_data, est_time, gt_data, gt_time, filter_data)

    plt.show()


def rmse_loss(est_data, est_time, gt_data, gt_time, filter_data):
    """
    Compute and plot the RMSE loss between the ground truth and the estimated data and the filtered data.
    :param est_data: Data estimated from camera data
    :param est_time: Times for estimated data
    :param gt_data: The ground truth data
    :param gt_time: Times for ground truth data
    :param filter_data: Data filtered by the Kalman Filter
    :return: ax
    """
    rmse_est = np.zeros(len(est_data[0]))
    rmse_filter = np.zeros(len(filter_data[0]))

    for i in range(len(est_data[0])):
        data_idx_gt = np.argmin(np.abs(gt_time - est_time[i]))
        rmse_est[i] = np.sqrt(np.mean((est_data[:3, i] - gt_data[:3, data_idx_gt]) ** 2))
        rmse_filter[i] = np.sqrt(np.mean((filter_data[:3, i] - gt_data[:3, data_idx_gt]) ** 2))

    # print("RMSE for Estimated Data: ", np.mean(rmse_est))
    # print("RMSE for Filtered Data: ", np.mean(rmse_filter))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(est_time, rmse_est, label='RMSE Estimated')
    ax.plot(est_time, rmse_filter, label='RMSE Filtered')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Loss')
    return ax


def task_2(filename):
    """
    Task 2: Plot the estimated position and orientation of the drone and compare with the ground truth data.
    :param filename: The filename of the data
    :return:
    """
    model = obs_model.ObsModel()
    data_list, gt_data, gt_time = model.process_data(filename)

    positions = []
    orientations = []
    est_time = []
    data = data_list
    for idx, dat in enumerate(data):
        if isinstance(dat['id'], int):
            pos, rot = model.estimate_pose([dat['id']], idx)
            positions.append(pos)
            orientations.append(rot)
            est_time.append(float(dat['t']))
            continue
        if dat['id'].shape[0] > 1:
            pos, rot = model.estimate_pose(dat['id'], idx)
            positions.append(pos)
            orientations.append(rot)
            est_time.append(float(dat['t']))

    est_data = np.zeros((6, len(positions)))
    est_data[0] = np.array([pos[0] for pos in positions])
    est_data[1] = np.array([pos[1] for pos in positions])
    est_data[2] = np.array([pos[2] for pos in positions])
    est_data[3] = np.array([rot[0] for rot in orientations])
    est_data[4] = np.array([rot[1] for rot in orientations])
    est_data[5] = np.array([rot[2] for rot in orientations])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.plot(gt_data[0], gt_data[1], gt_data[2], c='g', label='Ground Truth')
    ax.plot(est_data[0], est_data[1], est_data[2], 'r--', label='Estimated')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot for Position')

    # Plot to compare estimated position with ground truth
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(gt_time, gt_data[0])
    ax[0].plot(est_time, est_data[0])
    ax[0].set_ylabel('X')
    # ax[0].legend()

    ax[1].plot(gt_time, gt_data[1])
    ax[1].plot(est_time, est_data[1])
    ax[1].set_ylabel('Y')
    # ax[1].legend()

    ax[2].plot(gt_time, gt_data[2], label='Ground Truth')
    ax[2].plot(est_time, est_data[2], label='Estimated')
    ax[2].set_ylabel('Z')
    ax[2].set_xlabel('Time')
    ax[2].legend()
    ax[0].set_title('Position vs Time')

    # Plot to compare estimated angle with ground truth
    fig, ax = plt.subplots(3, 1)
    # ax[0].set_ylim(-np.pi / 2, np.pi / 2)
    ax[2].plot(gt_time, gt_data[5], label='Ground Truth')
    ax[2].plot(est_time, est_data[3], label='Estimated')
    ax[2].set_ylabel('Yaw')
    ax[2].legend()

    # ax[1].set_ylim(-np.pi / 2, np.pi / 2)
    ax[1].plot(gt_time, gt_data[4])
    ax[1].plot(est_time, est_data[4])
    ax[1].set_ylabel('Pitch')
    # ax[1].legend()

    # ax[2].set_ylim(-np.pi / 2, np.pi / 2)
    ax[0].plot(gt_time, gt_data[3])
    ax[0].plot(est_time, est_data[5])
    ax[0].set_ylabel('Roll')
    ax[0].set_title('Orientation vs Time')
    # ax[2].legend()

    plt.show()

