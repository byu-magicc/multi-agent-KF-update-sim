import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_xy_trajectories(poses=[], covariances=[], markers=[], lines=[]):
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
        cov = covariance[0]
        mean = covariance[1]
        eigvals, eigvecs = np.linalg.eig(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                      width=3 * np.sqrt(eigvals[0]),
                      height=3 * np.sqrt(eigvals[1]),
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
