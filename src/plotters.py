import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_overview(poses=[], covariances=[], markers=[], lines=[]):
    """
    Plots the top-down view of various agent trajectories, covariance elipses, global measurement
    markers, and range markers.

    Parameters:
    poses: Dynamically sized list of other lists containing these things: 2xn numpy array of xy
        poses, string name of trajectory (optional, pass "" for no name), color char for
        matplotlib (optional, pass "" for default color)
    covariances: Dynamically sized list of other lists containing these things: 2x2 numpy array of
        covariance, 2x1 numpy array of mean, color char for matplotlib (optional, pass "" for
        default color)
    markers: Dynamically sized list of other lists containing these things: 2x1 numpy array of xy
        positions for markers, color char for matplotlib (optional, pass "" for default color)
    lines: Dynamically sized list of other lists containing these things: 2x2 numpy array of xy
        endpoints for lines, color char for matplotlib (optional, pass "" for default color)
    """

    plt.figure()

    # Plot range markers
    for line in lines:
        assert line[0].shape == (2, 2)

        if line[1] == "":
            plt.plot(line[0][0, :], line[0][1, :], color='grey', linewidth=0.75)
        else:
            plt.plot(line[0][0, :], line[0][1, :], color=line[1], linewidth=0.75)

    # Plot trajectories
    for pose in poses:
        assert pose[0].shape[0] == 2
        assert pose[0].ndim == 2

        if pose[2] == "":
            plt.plot(pose[0][0, :], pose[0][1, :], label=pose[1])
        else:
            plt.plot(pose[0][0, :], pose[0][1, :], label=pose[1], color=pose[2])

    # Plot covariance ellipses
    for covariance in covariances:
        assert covariance[0].shape == (2, 2)
        assert covariance[1].shape == (2, 1)

        cov = covariance[0]
        mean = covariance[1].flatten()
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                      width=2 * np.sqrt(eigvals[0]) * 2,
                      height=2 * np.sqrt(eigvals[1]) * 2,
                      angle=np.rad2deg(angle),
                      edgecolor='b' if covariance[2] == "" else covariance[2],
                      facecolor='none',
                      zorder=4)
        plt.gca().add_artist(ell)

    # Plot global measurement markers
    for marker in markers:
        assert marker[0].shape == (2, 1)

        if marker[1] == "":
            plt.plot(marker[0][0, :], marker[0][1, :], 'x', color='r', markersize=8)
        else:
            plt.plot(marker[0][0, :], marker[0][1, :], 'x', color=marker[1], markersize=8)

    plt.legend()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_trajectory_error(mu_hist, truth_hist, Sigma_hist):
    """
    Plots the individual components of the error in the trajectory estimate, with the covariance
    bound.

    Parameters:
    mu_hist: Dictionary of nx5 numpy arrays of the state estimate for every timestep, where the key
        is the vehicle name.
    truth_hist: Dictionary of nx5 numpy arrays of the ground truth for every timestep, where the
        key is the vehicle name.
    Sigma_hist: Dictionary of nx5 numpy arrays of the standard deviation for every timestep, where
        the key is the vehicle name.
    """

    fig, axs = plt.subplots(3, len(mu_hist.keys()), figsize=(16, 12))

    if len(mu_hist.keys()) == 1:
        axs = np.expand_dims(axs, axis=1)

    column_idx = 0
    for key in mu_hist.keys():
        assert key in truth_hist.keys()
        assert key in Sigma_hist.keys()
        assert mu_hist[key].shape == truth_hist[key].shape
        assert mu_hist[key].shape == Sigma_hist[key].shape
        assert truth_hist[key].shape == Sigma_hist[key].shape

        error = mu_hist[key] - truth_hist[key]

        # X position
        axs[0, column_idx].plot(error[0, :], label='x error', color='r')
        axs[0, column_idx].plot(2*Sigma_hist[key][0, :], label='2 sigma', color='b')
        axs[0, column_idx].plot(-2*Sigma_hist[key][0, :], color='b')
        axs[0, column_idx].set_title(key)
        axs[0, column_idx].grid()

        # Y position
        axs[1, column_idx].plot(error[1, :], label='y error', color='r')
        axs[1, column_idx].plot(2*Sigma_hist[key][1, :], label='2 sigma', color='b')
        axs[1, column_idx].plot(-2*Sigma_hist[key][1, :], color='b')
        axs[1, column_idx].grid()

        # Psi
        axs[2, column_idx].plot(error[2, :], label='psi error', color='r')
        axs[2, column_idx].plot(2*Sigma_hist[key][2, :], label='2 sigma', color='b')
        axs[2, column_idx].plot(-2*Sigma_hist[key][2, :], color='b')
        axs[2, column_idx].grid()

        ## Vx
        #axs[3, column_idx].plot(error[3, :], label='vx error', color='r')
        #axs[3, column_idx].plot(2*Sigma_hist[key][3, :], label='2 sigma', color='b')
        #axs[3, column_idx].plot(-2*Sigma_hist[key][3, :], color='b')
        #axs[3, column_idx].grid()

        ## Vy
        #axs[4, column_idx].plot(error[4, :], label='vy error', color='r')
        #axs[4, column_idx].plot(2*Sigma_hist[key][4, :], label='2 sigma', color='b')
        #axs[4, column_idx].plot(-2*Sigma_hist[key][4, :], color='b')
        #axs[4, column_idx].grid()

        # Add ylabels and legend
        if column_idx == 0:
            axs[0, column_idx].set_ylabel('x error')
            axs[1, column_idx].set_ylabel('y error')
            axs[2, column_idx].set_ylabel('psi error')
            #axs[3, column_idx].set_ylabel('vx error')
            #axs[4, column_idx].set_ylabel('vy error')
            axs[0, column_idx].legend()

        column_idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from trajectories import sine_trajectory


    poses = [
        [sine_trajectory(100, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 10, 5)[:2, :], "", "b"],
        [sine_trajectory(100, np.array([[0, 0]]).T, np.array([[100, 0]]).T, 10, 5)[:2, :], "Traj", ""]
    ]

    lines = [
        [np.array([[0, 0], [100, 100]]).T, ""],
        [np.array([[0, 0], [100, 0]]).T, "r"]
    ]

    markers = [
        [np.array([[50], [50]]), "g"],
        [np.array([[50], [50]]), ""]
    ]

    covariances = [
        [np.array([[10, 0], [10, 1]]), np.array([[20], [50]]), "b"],
        [np.array([[12, -7], [-7, 6]]), np.array([[50], [20]]), "k"]
    ]

    plot_overview(poses=poses, lines=lines, markers=markers, covariances=covariances)

