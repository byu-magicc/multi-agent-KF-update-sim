import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import List, Dict


class Trajectory:
    """
    Struct for plot_overview function, containing all pose information and plotting parameters.
    """
    def __init__(self, pose_array: np.ndarray, name: str = "",
                 color: str = "b", opacity: float = 1.0):
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
    def __init__(self, covariance_array: np.ndarray, mean: np.ndarray,
                 name: str = "", color:str = "b"):
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
    def __init__(self, location: np.ndarray, color:str = "b"):
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
    def __init__(self, endpoints: np.ndarray, color: str = "b"):
        """
        endpoints: np.array (2, 2)
            The start and end points of the line. [[x1, x2], [y1, y2]]
        color: string
            MatPlotLib color key, specifying the color to plot the line.
        """
        assert endpoints.shape == (2, 2)
        self.endpoints = endpoints
        self.color = color


def plot_overview(trajectories: List[Trajectory] = [],
                  covariances: List[Covariance] = [],
                  markers: List[Markers] = [],
                  lines: List[Lines] = [],
                  num_sigma: int = 2):
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
        plt.plot(line.endpoints[0, :], line.endpoints[1, :], color=line.color, linewidth=0.75)

    # Plot trajectories
    for pose in trajectories:
        plt.plot(pose.poses[0, :], pose.poses[1, :], label=pose.name, color=pose.color,
                 alpha=pose.opacity)

    # Plot covariance ellipses
    for covariance in covariances:
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
        plt.plot(marker.location[0, :], marker.location[1, :], 'x', color=marker.color, markersize=8)

    plt.legend()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_trajectory_error(mu_hist: Dict[str, List[np.ndarray]],
                          truth_hist: Dict[str, List[np.ndarray]],
                          Sigma_hist: Dict[str, List[np.ndarray]],
                          num_sigma: int = 2,
                          sigma_only: bool = False):
    """
    Plots the individual components of the error in the trajectory estimate, with the covariance
    bound.

    Parameters:
    mu_hist: Dictionary of np.arrays (5 x n)
        State estimate for every timestep, where the key is the vehicle name.
    truth_hist: Dictionary of np.arrays (5 x n)
        Ground truth for every timestep, where the key is the vehicle name.
    Sigma_hist: Dictionary of np.arrays (n x 5 x 5)
        Covariance of estimate for every timestep, where the key is the vehicle name.
    num_sigma: int
        Number of standard deviations to plot for covariance.
    sigma_only: bool
        If True, only plot the covariance bound. Useful for very large numbers of trajectories
    """

    # Check inputs
    for key in mu_hist.keys():
        assert key in truth_hist.keys()
        assert key in Sigma_hist.keys()
        assert len(mu_hist[key]) == len(truth_hist[key])
        assert len(mu_hist[key]) == len(Sigma_hist[key])

        for i in range(len(mu_hist[key])):
            assert mu_hist[key][i].shape[0] == 5
            assert mu_hist[key][i].ndim == 2
            assert mu_hist[key][i].shape == truth_hist[key][i].shape
            assert mu_hist[key][i].shape[1] == Sigma_hist[key][i].shape[0]
            assert Sigma_hist[key][i].shape[1:] == (5, 5)

    # Check if we have multiple instances
    overlay_plots = len(next(iter(mu_hist.values()))) > 1
    alpha = 0.5 if overlay_plots else 1.0

    # Calculate population standard deviation
    calculated_Sigma = {}
    if overlay_plots:
        for key in mu_hist.keys():
            residuals = []
            for i in range(len(mu_hist[key])):
                residuals.append((mu_hist[key][i] - truth_hist[key][i])**2)
            residuals = np.array(residuals)
            calculated_Sigma[key] = np.sqrt(np.mean(residuals, axis=0))

    # Covariance only plot
    if sigma_only:
        fig, axs = plt.subplots(3, len(mu_hist.keys()), figsize=(16, 12))
        column_idx = 0
        for key in mu_hist.keys():
            curr_Sigma = Sigma_hist[key][0][:, :3, :3]
            curr_Sigma = np.hstack([
                np.sqrt(Sigma.diagonal().reshape(-1, 1)) for Sigma in curr_Sigma
            ])

            # X
            axs[0, column_idx].plot(curr_Sigma[0, :], label='Estimator Sigma', color='b')
            axs[0, column_idx].plot(calculated_Sigma[key][0, :], label='Actual Sigma', color='g')
            axs[0, column_idx].set_title(key)
            axs[0, column_idx].grid()

            # Y
            axs[1, column_idx].plot(curr_Sigma[1, :], color='b')
            axs[1, column_idx].plot(calculated_Sigma[key][1, :], color='g')
            axs[1, column_idx].grid()

            # Psi
            axs[2, column_idx].plot(curr_Sigma[2, :], color='b')
            axs[2, column_idx].plot(calculated_Sigma[key][2, :], color='g')
            axs[2, column_idx].grid()

            # Add ylabels and legend
            if column_idx == 0:
                axs[0, column_idx].set_ylabel('X Error (m)')
                axs[1, column_idx].set_ylabel('Y Error (m)')
                axs[2, column_idx].set_ylabel('Psi Error (rad)')
                axs[0, column_idx].legend()

            column_idx += 1

        plt.tight_layout()
        plt.show()
        return

    # Create full plot
    fig, axs = plt.subplots(3, len(mu_hist.keys()), figsize=(16, 12))
    if len(mu_hist.keys()) == 1:
        axs = np.expand_dims(axs, axis=1)

    column_idx = 0
    for key in mu_hist.keys():
        for i in range(len(mu_hist[key])):
            curr_mu = mu_hist[key][i]
            curr_truth = truth_hist[key][i]
            curr_Sigma = Sigma_hist[key][i]

            # Condense mu, truth, and Sigma to just the x, y, psi values
            # Temporary fix, as we don't have vx and vy truth values for error plotting
            curr_mu = curr_mu[:3, :]
            curr_truth = curr_truth[:3, :]
            curr_Sigma = curr_Sigma[:, :3, :3]

            # Get sigma values for state variables
            curr_Sigma = np.hstack([
                np.sqrt(Sigma.diagonal().reshape(-1, 1)) for Sigma in curr_Sigma
            ])

            error = curr_mu - curr_truth

            # X position
            if i == 0:
                axs[0, column_idx].plot(error[0, :], label='Error', color='r', alpha=alpha)
                axs[0, column_idx].plot(num_sigma*curr_Sigma[0, :],
                                        label=f'Estimator {num_sigma} Sigma',
                                        color='b')
                axs[0, column_idx].plot(-num_sigma*curr_Sigma[0, :], color='b')
                axs[0, column_idx].set_title(key)
                axs[0, column_idx].grid()

                if overlay_plots:
                    axs[0, column_idx].plot(num_sigma*calculated_Sigma[key][0, :],
                                            label=f'Calculated {num_sigma} Sigma', color='g')
                    axs[0, column_idx].plot(-num_sigma*calculated_Sigma[key][0, :], color='g')
            else:
                axs[0, column_idx].plot(error[0, :], color='r', alpha=alpha)

            # Y position
            axs[1, column_idx].plot(error[1, :], color='r', alpha=alpha)
            if i == 0:
                axs[1, column_idx].plot(num_sigma*curr_Sigma[1, :], color='b')
                axs[1, column_idx].plot(-num_sigma*curr_Sigma[1, :], color='b')
                axs[1, column_idx].grid()

                if overlay_plots:
                    axs[1, column_idx].plot(num_sigma*calculated_Sigma[key][1, :],
                                label=f'Actual {num_sigma} Sigma', color='g')
                    axs[1, column_idx].plot(-num_sigma*calculated_Sigma[key][1, :], color='g')

            # Psi
            axs[2, column_idx].plot(error[2, :], color='r', alpha=alpha)
            if i == 0:
                axs[2, column_idx].plot(num_sigma*curr_Sigma[2, :], color='b')
                axs[2, column_idx].plot(-num_sigma*curr_Sigma[2, :], color='b')
                axs[2, column_idx].grid()

                if overlay_plots:
                    axs[2, column_idx].plot(num_sigma*calculated_Sigma[key][2, :],
                    label=f'Actual {num_sigma} Sigma', color='g')
                    axs[2, column_idx].plot(-num_sigma*calculated_Sigma[key][2, :], color='g')

            # Add ylabels and legend
            if column_idx == 0:
                axs[0, column_idx].set_ylabel('X Error (m)')
                axs[1, column_idx].set_ylabel('Y Error (m)')
                axs[2, column_idx].set_ylabel('Psi Error (rad)')
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

