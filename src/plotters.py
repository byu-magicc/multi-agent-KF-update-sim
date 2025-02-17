import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class Trajectory:
    """
    Struct for plot_overview function, containing all pose information and plotting parameters.
    """
    def __init__(self, pose_array, name="", color="b", opacity=1.0):
        """
        pose_array: np.array (2, n)
            Array of all poses that a vehicle traveled during a trajectory.
        name: string
            Name to give trajectory in legend.
        color: string
            MatPlotLib color key, specifying the color to plot the trajectory.
        opacity: float, 0.0 <= opacity <=1.0
            Opacity to plot trajectory with.
        """
        assert pose_array.shape[0] == 2
        assert pose_array.ndim == 2
        assert opacity >= 0.0
        assert opacity <= 1.0

        self.poses = pose_array
        self.name = name
        self.color = color
        self.opacity = opacity


class Covariance:
    """
    Struct for plot_overview function, containing all Covariance ellipse information and plotting
    parameters.
    """
    def __init__(self, covariance_array, mean, name="", color="b"):
        """
        covariance_array: np.array (2, 2)
            Covariance of ellipse.
        mean: np.array (2, 1)
            Location to center ellipse at.
        name: string
            Name to give ellipse in legend.
        color: string
            MatPlotLib color key, specifying the color to plot the ellipse.
        """
        assert covariance_array.shape == (2, 2)
        assert mean.shape == (2, 1)

        self.covariance_array = covariance_array
        self.mean = mean
        self.name = name
        self.color = color


class Markers:
    """
    Struct for plot_overview function, containing all marker information and plotting parameters.
    """
    def __init__(self, location, color="b"):
        """
        location: np.array (2, 1)
            Location to center marker at.
        color: string
            MatPlotLib color key, specifying the color to plot the marker.
        """
        assert location.shape == (2, 1)

        self.location = location
        self.color = color


class Lines:
    """
    Struct for plot_overview function, containing all line information and plotting parameters.
    """
    def __init__(self, endpoints, color="b"):
        """
        endpoints: np.array (2, 2)
            The start and end points of the line. [[x1, x2], [y1, y2]]
        color: string
            MatPlotLib color key, specifying the color to plot the line.
        """
        assert endpoints.shape == (2, 2)
        self.endpoints = endpoints
        self.color = color


def plot_overview(trajectories = [], covariances = [], markers = [], lines = [], num_sigma=2):
    """
    Plots the top-down view of various agent trajectories, covariance elipses, global measurement
    markers, and range markers.

    Parameters:
    trajectories: list of Trajectory classes
        Trajectories to plot.
    covariance: list of Covariance classes
        Covariance ellipses to plot.
    markers: list of Markers classes
        Global measurement markers to plot.
    lines: list of Lines classes
        Range measurements to plot as lines.
    num_sigma: int
        Number of standard deviations to plot for covariance ellipses.
    """
    assert num_sigma > 0

    plt.figure()

    # Plot range markers
    for line in lines:
        assert isinstance(line, Lines)
        plt.plot(line.endpoints[0, :], line.endpoints[1, :], color=line.color, linewidth=0.75)

    # Plot trajectories
    for pose in trajectories:
        assert isinstance(pose, Trajectory)
        plt.plot(pose.poses[0, :], pose.poses[1, :], label=pose.name, color=pose.color,
                 alpha=pose.opacity)

    # Plot covariance ellipses
    for covariance in covariances:
        assert isinstance(covariance, Covariance)

        cov = covariance.covariance_array
        mean = covariance.mean.flatten()
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                      width=num_sigma * np.sqrt(eigvals[0]) * 2,
                      height=num_sigma * np.sqrt(eigvals[1]) * 2,
                      angle=np.rad2deg(angle),
                      edgecolor=covariance.color,
                      facecolor="none",
                      zorder=4,
                      label=covariance.name
                      )
        plt.gca().add_artist(ell)

    # Plot global measurement markers
    for marker in markers:
        assert isinstance(marker, Markers)

        plt.plot(marker.location[0, :], marker.location[1, :], 'x', color=marker.color, markersize=8)

    plt.legend()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_trajectory_error(mu_hist, truth_hist, Sigma_hist, num_sigma=2):
    """
    Plots the individual components of the error in the trajectory estimate, with the covariance
    bound.

    Parameters:
    mu_hist: Dictionary of 5xn numpy arrays of the state estimate for every timestep, where the key
        is the vehicle name.
    truth_hist: Dictionary of 5xn numpy arrays of the ground truth for every timestep, where the
        key is the vehicle name.
    Sigma_hist: Dictionary of nx5x5 numpy arrays of the standard deviation for every timestep,
        where the key is the vehicle name.
    num_sigma: Number of standard deviations to plot for covariance.
    """

    fig, axs = plt.subplots(3, len(mu_hist.keys()), figsize=(16, 12))

    if len(mu_hist.keys()) == 1:
        axs = np.expand_dims(axs, axis=1)

    column_idx = 0
    for key in mu_hist.keys():
        assert key in truth_hist.keys()
        assert key in Sigma_hist.keys()
        assert mu_hist[key].shape[0] == 5
        assert mu_hist[key].shape == truth_hist[key].shape
        assert mu_hist[key].shape[1] == Sigma_hist[key].shape[0]
        assert Sigma_hist[key].shape[1:] == (5, 5)

        # Condense mu_hist, truth_hist, and Sigma_hist to just the x, y, psi values
        # Temporary fix, as we don't have vx and vy truth values for error plotting
        mu_hist[key] = mu_hist[key][:3, :]
        truth_hist[key] = truth_hist[key][:3, :]
        Sigma_hist[key] = Sigma_hist[key][:, :3, :3]

        # Get sigma values for state variables
        Sigma_hist[key] = np.hstack([
            np.sqrt(Sigma.diagonal().reshape(-1, 1)) for Sigma in Sigma_hist[key]
        ])

        error = mu_hist[key] - truth_hist[key]

        # X position
        axs[0, column_idx].plot(error[0, :], label='Error', color='r')
        axs[0, column_idx].plot(num_sigma*Sigma_hist[key][0, :],
                                label=f'{num_sigma} Sigma',
                                color='b')
        axs[0, column_idx].plot(-num_sigma*Sigma_hist[key][0, :], color='b')
        axs[0, column_idx].set_title(key)
        axs[0, column_idx].grid()

        # Y position
        axs[1, column_idx].plot(error[1, :], color='r')
        axs[1, column_idx].plot(num_sigma*Sigma_hist[key][1, :], color='b')
        axs[1, column_idx].plot(-num_sigma*Sigma_hist[key][1, :], color='b')
        axs[1, column_idx].grid()

        # Psi
        axs[2, column_idx].plot(error[2, :], color='r')
        axs[2, column_idx].plot(num_sigma*Sigma_hist[key][2, :], color='b')
        axs[2, column_idx].plot(-num_sigma*Sigma_hist[key][2, :], color='b')
        axs[2, column_idx].grid()

        ## Vx
        #axs[3, column_idx].plot(error[3, :], color='r')
        #axs[3, column_idx].plot(num_sigma*Sigma_hist[key][3, :], color='b')
        #axs[3, column_idx].plot(-num_sigma*Sigma_hist[key][3, :], color='b')
        #axs[3, column_idx].grid()

        ## Vy
        #axs[4, column_idx].plot(error[4, :], color='r')
        #axs[4, column_idx].plot(num_sigma*Sigma_hist[key][4, :], color='b')
        #axs[4, column_idx].plot(-num_sigma*Sigma_hist[key][4, :], color='b')
        #axs[4, column_idx].grid()

        # Add ylabels and legend
        if column_idx == 0:
            axs[0, column_idx].set_ylabel('X Error')
            axs[1, column_idx].set_ylabel('Y Error')
            axs[2, column_idx].set_ylabel('Psi Error')
            #axs[3, column_idx].set_ylabel('Vx Error')
            #axs[4, column_idx].set_ylabel('Vy Error')
            axs[0, column_idx].legend()

        column_idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from trajectories import sine_trajectory


    poses = [
        Trajectory(
            sine_trajectory(100, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 10, 5)[:2, :]
        ),
        Trajectory(
            sine_trajectory(100, np.array([[0, 0]]).T, np.array([[100, 0]]).T, 10, 5)[:2, :],
            name="Traj"
        )
    ]

    lines = [
        Lines(
            np.array([[0, 0], [100, 100]]).T
        ),
        Lines(
            np.array([[0, 0], [100, 0]]).T,
            color="r"
        )
    ]

    markers = [
        Markers(np.array([[50], [50]]), color="g"),
        Markers(np.array([[50], [50]]))
    ]

    covariances = [
        Covariance(
            np.array([[10, 0], [0, 1]]),
            np.array([[20], [50]]),
            color="b"
        ),
        Covariance(
            np.array([[12, -7], [-7, 6]]),
            np.array([[50], [20]]),
            color="k"
        )
    ]

    plot_overview(trajectories=poses, lines=lines, markers=markers, covariances=covariances)

