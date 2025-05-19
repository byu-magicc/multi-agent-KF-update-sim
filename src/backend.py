from functools import partial
from typing import List, Optional

import gtsam
import numpy as np


class Prior:
    """
    Struct for storing priors.
    """
    def __init__(self, vehicle: str, mean: np.ndarray, sigmas: np.ndarray):
        """
        vehicle: str
            The name of the vehicle.
        mean: np.ndarray (3, 1)
            The mean of the prior. x, y, theta.
        sigmas: np.ndarray (3, 1)
            The standard deviations of the prior.
        """
        assert mean.shape == (3, 1)
        assert sigmas.shape == (3, 1)

        self.vehicle = vehicle
        self.mean = mean
        self.sigmas = sigmas


class Odometry:
    """
    Struct for storing odometry measurements.
    """
    def __init__(self, vehicle: str, mean: np.ndarray, Sigmas: np.ndarray):
        """
        vehicle: str
            The name of the vehicle.
        mean: np.ndarray (3, 1)
            The mean of the odometry. x, y, theta.
        Sigmas: np.ndarray (3, 1)
            The standard deviations of the odometry.
        """
        assert mean.shape == (3, 1)
        assert Sigmas.shape == (3, 1)

        self.vehicle = vehicle
        self.mean = mean
        self.Sigmas = Sigmas


class Global:
    """
    Struct for storing global measurements.
    """
    def __init__(self, vehicle: str, mean: np.ndarray, Sigmas: np.ndarray):
        """
        vehicle: str
            The name of the vehicle.
        mean: np.ndarray (3, 1)
            The mean of the global measurement. x, y, theta.
        Sigmas: np.ndarray (3, 1)
            The standard deviations of the global measurement.
        """
        assert mean.shape == (3, 1)
        assert Sigmas.shape == (3, 1)

        self.vehicle = vehicle
        self.mean = mean
        self.Sigmas = Sigmas


class Range:
    """
    Struct for storing range measurements.
    """
    def __init__(self, vehicle1: str, vehicle2: str, mean: float, Sigma: float):
        """
        vehicle1: str
            The name of the first associated vehicle.
        vehicle2: str
            The name of the second associated vehicle.
        mean: float
            The mean of the range measurement.
        Sigmas: float
            The standard deviation of the range measurement.
        """
        self.vehicle1 = vehicle1
        self.vehicle2 = vehicle2
        self.mean = mean
        self.Sigma = Sigma


class Backend:
    """
    Multi-agent backend, utilizing GTSAM for factor graph optimization.
    """
    def __init__(self, priors: List[Prior]):
        # Check for duplicate vehicle names
        vehicles = [prior.vehicle for prior in priors]
        if len(vehicles) != len(set(vehicles)):
            raise ValueError("Duplicate vehicle names in priors.")

        # Create gtsam components
        self.graph = gtsam.NonlinearFactorGraph()
        self.params = gtsam.LevenbergMarquardtParams()

        # Add the priors, keeping track of the vehicle pose ids
        self.next_id = 0
        self.vehicle_pose_ids = {}
        self.current_estimates = gtsam.Values()
        for prior in priors:
            self.graph.add(
                gtsam.PriorFactorPose2(
                    self.next_id,
                    gtsam.Pose2(*prior.mean),
                    gtsam.noiseModel.Diagonal.Sigmas(prior.sigmas.flatten())
                )
            )
            self.current_estimates.insert(self.next_id, gtsam.Pose2(*prior.mean))
            self.vehicle_pose_ids[prior.vehicle] = [self.next_id]
            self.next_id += 1

        self.graph_outdated = True

    def add_odometry(self, odometry: Odometry):
        """
        Add an odometry measurement to the graph for the specified vehicle.

        odometry: Odometry
            The odometry measurement to add.
        """

        # Check the vehicle name already exists
        if odometry.vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {odometry.vehicle} not found in priors.")

        # Add the odometry factor
        self.graph_outdated = True
        self.graph.add(
            gtsam.BetweenFactorPose2(
                self.vehicle_pose_ids[odometry.vehicle][-1],
                self.next_id,
                gtsam.Pose2(*odometry.mean),
                gtsam.noiseModel.Diagonal.Sigmas(odometry.Sigmas.flatten())
            )
        )

        # Add a new estimate to the current estimates
        x = self.current_estimates.atPose2(self.vehicle_pose_ids[odometry.vehicle][-1]).x()
        y = self.current_estimates.atPose2(self.vehicle_pose_ids[odometry.vehicle][-1]).y()
        theta = self.current_estimates.atPose2(self.vehicle_pose_ids[odometry.vehicle][-1]).theta()
        delta_x = odometry.mean.item(0) * np.cos(theta) - odometry.mean.item(1) * np.sin(theta)
        delta_y = odometry.mean.item(0) * np.sin(theta) + odometry.mean.item(1) * np.cos(theta)
        delta_theta = odometry.mean.item(2)
        self.current_estimates.insert(
            self.next_id, gtsam.Pose2(x + delta_x, y + delta_y, theta + delta_theta)
        )

        # Add the new pose id to the vehicle's list
        self.vehicle_pose_ids[odometry.vehicle].append(self.next_id)
        self.next_id += 1

    def add_global(self, global_measurement: Global):
        """
        Add a global measurement to the graph for the specified vehicle.

        global_measurement: Global
            The global measurement to add.
        """

        # Check the vehicle name already exists
        if global_measurement.vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {global_measurement.vehicle} not found in priors.")

        # Add the global factor
        self.graph_outdated = True
        self.graph.add(
            gtsam.CustomFactor(
                gtsam.noiseModel.Diagonal.Sigmas(global_measurement.Sigmas.flatten()),
                [self.vehicle_pose_ids[global_measurement.vehicle][-1]],
                partial(_error_global, global_measurement.mean.flatten())
            )
        )

    def add_range(self, range_measurement: Range):
        """
        Add a range measurement to the graph for the specified vehicles.

        range_measurement: Range
            The range measurement to add.
        """

        # Check the vehicle names already exist
        if range_measurement.vehicle1 not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {range_measurement.vehicle1} not found in priors.")
        if range_measurement.vehicle2 not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {range_measurement.vehicle2} not found in priors.")

        # Add the range factor
        self.graph_outdated = True
        self.graph.add(
            gtsam.RangeFactorPose2(
                self.vehicle_pose_ids[range_measurement.vehicle1][-1],
                self.vehicle_pose_ids[range_measurement.vehicle2][-1],
                range_measurement.mean,
                gtsam.noiseModel.Isotropic.Sigma(1, range_measurement.Sigma)
            )
        )

    def get_vehicle_info(self, vehicle: str):
        """
        Optimizes the graph and returns the estimated pose and uncertainty for the specified
        vehicle.

        vehicle: str
            The name of the vehicle.

        Return: (np.ndarray(3,1), np.ndarray(3,3))
            The estimated pose and uncertainty of the vehicle.
        """

        # Check the vehicle name exists
        if vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {vehicle} not found in priors.")

        # Update the graph and return the estimated pose and uncertainty
        self._update()
        pose = np.array([
            self.current_estimates.atPose2(self.vehicle_pose_ids[vehicle][-1]).x(),
            self.current_estimates.atPose2(self.vehicle_pose_ids[vehicle][-1]).y(),
            self.current_estimates.atPose2(self.vehicle_pose_ids[vehicle][-1]).theta()
        ]).reshape(-1, 1)

        # Get the covariance of the last pose in global frame
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        covariance = marginals.marginalCovariance(self.vehicle_pose_ids[vehicle][-1])
        theta = self.current_estimates.atPose2(self.vehicle_pose_ids[vehicle][-1]).theta()
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0,              0,             1]])
        covariance = rot @ covariance @ rot.T

        return pose, covariance

    def get_full_trajectory(self, vehicle: str):
        """
        Optimizes the graph and returns the full trajectory for the specified vehicle.

        vehicle: str
            The name of the vehicle.

        Return: (np.ndarray(3,n), np.ndarray(n,3,3))
            poses and uncertainty, in global (not local) frame
        """

        # Check the vehicle name exists
        if vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {vehicle} not found in priors.")

        # Update the graph and get full trajectory
        self._update()
        poses = np.array([
            [self.current_estimates.atPose2(i).x(),
             self.current_estimates.atPose2(i).y(),
             self.current_estimates.atPose2(i).theta()]
            for i in self.vehicle_pose_ids[vehicle]
        ]).T

        # Get the covariances in global frame
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        covariances = []
        for i in self.vehicle_pose_ids[vehicle]:
            theta = self.current_estimates.atPose2(i).theta()
            cov = marginals.marginalCovariance(i)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta),  np.cos(theta), 0],
                            [0,              0,             1]])
            cov = rot @ cov @ rot.T
            covariances.append(cov)
        covariances = np.array(covariances)

        return poses, covariances

    def get_transformation(self, vehicle_1: str, vehicle_2: str):
        """
        Get tranformation from vehicle_1 to vehicle_2 with the correct tranformational uncertainty.
        Returns the transformation of the latest poses, in global frame.

        Mirrors Brendon's partial_update_msckf repository.
        https://github.com/byu-magicc-archive/brendon__partial_update_msckf/blob/main/backend.py

        vehicle_1: str
            The name of the first vehicle.
        vehicle_2: str
            The name of the second vehicle.

        Returns: (np.ndarray(3, 1), (np.ndarray(3, 3))
            Transformation and covariance
        """

        if vehicle_1 not in self.vehicle_pose_ids or vehicle_2 not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {vehicle_1} or {vehicle_2} not found in priors.")

        self._update()

        # Get the latest poses
        pose_1 = self.current_estimates.atPose2(self.vehicle_pose_ids[vehicle_1][-1])
        pose_2 = self.current_estimates.atPose2(self.vehicle_pose_ids[vehicle_2][-1])
        t_1_2 = gtsam.Pose2.between(pose_1, pose_2)
        t_1_2 = np.array([t_1_2.x(), t_1_2.y(), t_1_2.theta()]).reshape(-1, 1)

        # Get the joint covariance of the two poses
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        joint_cov = marginals.jointMarginalCovariance([
            self.vehicle_pose_ids[vehicle_1][-1],
            self.vehicle_pose_ids[vehicle_2][-1]
        ]).fullMatrix()
        cov_1 = joint_cov[:3, :3]
        cov_2 = joint_cov[3:, 3:]
        cov_12 = joint_cov[:3, 3:]
        cov_21 = joint_cov[3:, :3]

        # Get the covariance of the transformation
        adj = pose_2.inverse().AdjointMap() @ pose_1.AdjointMap()
        cov = adj @ cov_1 @ adj.T + cov_2 - adj @ cov_12 - cov_21 @ adj.T

        # Rotate to the global frame
        theta = pose_1.theta()
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0,              0,             1]])
        t_1_2_global = R @ t_1_2

        cov_global = R @ cov @ R.T

        return t_1_2_global, cov_global


    def _update(self):
        """
        Update the graph if it is outdated.
        """
        if self.graph_outdated:
            self.current_estimates = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.current_estimates, self.params
            ).optimize()
            self.graph_outdated = False


def _error_global(measurement: np.ndarray,
                  this: gtsam.CustomFactor,
                  values: gtsam.Values,
                  jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """
    Global Factor error function

    measurement:
        Global measurement, to be filled with `partial`.
    this:
        gtsam.CustomFactor handle.
    values:
        gtsam.Values.
    jacobians: 
        Optional list of Jacobians.

    Return: the unwhitened error
    """
    key = this.keys()[0]
    estimate = values.atPose2(key)
    theta = estimate.theta()
    estimate = np.array([estimate.x(), estimate.y(), estimate.theta()])
    error = estimate - measurement

    if jacobians is not None:
        jacobians[0] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])

    return error


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse

    prior_noise = np.array([[0.1], [0.1], [0.1]])
    odometry_noise = np.array([[0.2], [0.2], [0.1]])
    global_noise = np.array([[0.1], [0.1], [np.inf]])
    range_noise = 0.1

    priors = [
        Prior("A", np.array([[0], [0], [0]]), prior_noise),
        Prior("B", np.array([[0], [5], [0]]), prior_noise),
    ]
    odometries_1 = [
        Odometry("A", np.array([[2], [0], [0]]), odometry_noise),
        Odometry("A", np.array([[2], [0], [0]]), odometry_noise),
        Odometry("B", np.array([[2], [0], [0]]), odometry_noise),
        Odometry("B", np.array([[2], [0], [0]]), odometry_noise),
    ]
    range_measurements = [
        Range("A", "B", 6, range_noise),
    ]
    odometries_2 = [
        Odometry("A", np.array([[2], [0], [0]]), odometry_noise),
        Odometry("A", np.array([[2], [0], [0]]), odometry_noise),
        Odometry("B", np.array([[2], [0], [0]]), odometry_noise),
        Odometry("B", np.array([[2], [0], [0]]), odometry_noise),
    ]
    globals = [
        Global("A", np.array([[8], [0], [0]]), global_noise),
    ]

    backend = Backend(priors)
    for odometry in odometries_1:
        backend.add_odometry(odometry)
    for range_meas in range_measurements:
        backend.add_range(range_meas)
    for odometry in odometries_2:
        backend.add_odometry(odometry)
    for global_measurement in globals:
        backend.add_global(global_measurement)

    poses_1, covariances_1 = backend.get_full_trajectory("A")
    poses_2, covariances_2 = backend.get_full_trajectory("B")

    print(backend.get_transformation("B", "A"))

    plt.figure()
    plt.plot(poses_1[0, :], poses_1[1, :], color="b", marker="o", label="A")
    plt.plot(poses_2[0, :], poses_2[1, :], color="g", marker="o", label="B")

    for i in range(poses_1.shape[1]):
        num_sigma = 2
        cov = covariances_1[i][:2, :2]
        mean = poses_1[:2, i]
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                      width=num_sigma * np.sqrt(eigvals[0]) * 2,
                      height=num_sigma * np.sqrt(eigvals[1]) * 2,
                      angle=np.rad2deg(angle),
                      edgecolor="r",
                      facecolor="none",
                      zorder=4,
                      )
        plt.gca().add_artist(ell)
    for i in range(poses_2.shape[1]):
        num_sigma = 2
        cov = covariances_2[i][:2, :2]
        mean = poses_2[:2, i]
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                  width=num_sigma * np.sqrt(eigvals[0]) * 2,
                  height=num_sigma * np.sqrt(eigvals[1]) * 2,
                  angle=np.rad2deg(angle),
                  edgecolor="r",
                  facecolor="none",
                  zorder=4,
                  )
        plt.gca().add_artist(ell)

    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()


