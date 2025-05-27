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
        mean: np.ndarray (5, 1)
            The mean of the prior. x, y, theta, vx, vy.
        sigmas: np.ndarray (5, 1)
            The standard deviations of the prior.
        """
        assert mean.shape == (5, 1)
        assert sigmas.shape == (5, 1)

        self.vehicle = vehicle

        self.position = np.zeros(3)
        self.position[:2] = mean[:2].flatten()

        self.attitude = np.zeros(3)
        self.attitude[-1] = mean[2]

        self.position_sigmas = np.ones(3) * 1e-6
        self.position_sigmas[:2] = sigmas[:2].flatten()

        self.attitude_sigmas = np.ones(3) * 1e-6
        self.attitude_sigmas[-1] = sigmas[2]

        self.velocity = np.zeros(3)
        self.velocity[:2] = mean[3:].flatten()

        self.velocity_sigmas = np.ones(3) * 1e-6
        self.velocity_sigmas[:2] = sigmas[3:].flatten()


class IMU:
    """
    Struct for storing IMU measurements.
    """
    def __init__(self, vehicle: str, mean: np.ndarray):
        """
        Params:
        vehicle: str
            The name of the vehicle.
        mean: np.ndarray(3,1)
            The mean of the IMU measurement (ax, ay, omega).
        """
        assert mean.shape == (3, 1)

        self.vehicle = vehicle

        self.accel = np.zeros(3)
        self.accel[:2] = mean[:2].flatten()

        self.gyro = np.zeros(3)
        self.gyro[-1] = mean[-1].flatten()


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
    def __init__(self, priors: List[Prior], imu_sigmas: np.ndarray, imu_dt: float):
        """
        Params:
        priors: List[Prior]
            List of prior for each vehicle. Determines the number of vehicles in the system.
        imu_sigmas: np.ndarray(3,1)
            The standard deviation of the IMU measurements (ax, ay, omega).
        imu_dt: float
            The time step for the IMU measurements.
        """
        assert imu_sigmas.shape == (3, 1)

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
        self.vehicle_velocity_ids = {}
        self.vehicle_bias_ids = {}
        self.current_estimates = gtsam.Values()
        for prior in priors:
            attitude = gtsam.Rot3.RzRyRx(*prior.attitude)
            position = gtsam.Point3(*prior.position)

            self.graph.add(
                gtsam.PriorFactorPose3(
                    self.next_id,
                    gtsam.Pose3(attitude, position),
                    gtsam.noiseModel.Diagonal.Sigmas(np.append(
                        prior.attitude_sigmas, prior.position_sigmas
                    ))
                )
            )
            self.current_estimates.insert(self.next_id, gtsam.Pose3(attitude, position))
            self.vehicle_pose_ids[prior.vehicle] = [self.next_id]
            self.next_id += 1

            self.graph.add(
                gtsam.PriorFactorVector(
                    self.next_id,
                    prior.velocity,
                    gtsam.noiseModel.Diagonal.Sigmas(prior.velocity_sigmas)
                )
            )
            self.current_estimates.insert(self.next_id, prior.velocity)
            self.vehicle_velocity_ids[prior.vehicle] = [self.next_id]
            self.next_id += 1

            bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
            self.graph.add(
                gtsam.PriorFactorConstantBias(
                    self.next_id,
                    bias,
                    gtsam.noiseModel.Isotropic.Sigma(6, 1e-9)
                )
            )
            self.current_estimates.insert(self.next_id, bias)
            self.vehicle_bias_ids[prior.vehicle] = [self.next_id]
            self.next_id += 1

        # IMU pre-integration objects
        pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU(1e-9)
        accel_Sigma = np.eye(3) * 1e-9
        accel_Sigma[0, 0] = imu_sigmas[0]**2
        accel_Sigma[1, 1] = imu_sigmas[1]**2
        gyro_Sigma = np.eye(3) * 1e-9
        gyro_Sigma[2, 2] = imu_sigmas[2] ** 2
        pim_params.setGyroscopeCovariance(gyro_Sigma)
        pim_params.setAccelerometerCovariance(accel_Sigma)
        pim_params.setIntegrationCovariance(1e-9 * np.eye(3))
        pim_params.setBiasAccCovariance(1e-9 * np.eye(3))
        pim_params.setBiasOmegaCovariance(1e-9 * np.eye(3))
        pim_params.setBiasAccOmegaInit(1e-9 * np.eye(6))
        self.vehicle_pims = {}
        for prior in priors:
            self.vehicle_pims[prior.vehicle] = gtsam.PreintegratedCombinedMeasurements(
                pim_params,
                gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
            )

        self.graph_outdated = True
        self.imu_dt = imu_dt

    # TODO: Update to 3d
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

    # TODO: Update to 3d
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
        Optimizes the graph and returns the estimated state and uncertainty for the specified
        vehicle.

        vehicle: str
            The name of the vehicle.

        Return: (np.ndarray(5,1), np.ndarray(5,5))
            The estimated state and uncertainty of the vehicle.
        """

        # Check the vehicle name exists
        if vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {vehicle} not found in priors.")

        # Update the graph and return the estimated pose and uncertainty
        self._create_imu_edge(vehicle)
        self._update()
        attitude = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).rotation()
        position = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).translation()
        velocity = self.current_estimates.atVector(self.vehicle_velocity_ids[vehicle][-1])
        x = position.item(0)
        y = position.item(1)
        theta = attitude.yaw()
        estimate = np.array([x, y, theta, velocity.item(0), velocity.item(1)]).reshape(-1, 1)

        # Get the covariance of the last pose in global frame
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        covariance = marginals.jointMarginalCovariance([
            self.vehicle_pose_ids[vehicle][-1],
            self.vehicle_velocity_ids[vehicle][-1],
        ]).fullMatrix()

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        R_full = np.eye(5)
        R_full[:2, :2] = R
        R_full[3:, 3:] = R
        M = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0]])

        cov_2d = R_full @ M @ covariance @ M.T @ R_full.T

        return estimate, cov_2d

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
        self._create_imu_edge(vehicle)
        self._update()
        estimates = []
        for i, j in zip(self.vehicle_pose_ids[vehicle], self.vehicle_velocity_ids[vehicle]):
            position = self.current_estimates.atPose3(i).translation()
            attitude = self.current_estimates.atPose3(i).rotation()
            velocity = self.current_estimates.atVector(j)
            estimates.append(np.array([[position.item(0), position.item(1), attitude.yaw(),
                                        velocity.item(0), velocity.item(1)]]).T)
        estimates = np.hstack(estimates)

        # Get the covariances in global frame
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        covariances = []
        M = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        for i, j in zip(self.vehicle_pose_ids[vehicle], self.vehicle_velocity_ids[vehicle]):
            cov = marginals.jointMarginalCovariance([i, j]).fullMatrix()

            theta = self.current_estimates.atPose3(i).rotation().yaw()
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            R_full = np.eye(5)
            R_full[:2, :2] = R
            R_full[3:, 3:] = R
            cov_2d = R_full @ M @ cov @ M.T @ R_full.T

            covariances.append(cov_2d)
        covariances = np.array(covariances)

        return estimates, covariances

    def add_imu(self, imu: IMU):
        """
        Add an IMU measurement to the graph for the specified vehicle.
        """

        # Check the vehicle name already exists
        if imu.vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {imu.vehicle} not found in priors.")

        # Add the IMU measurement to the pre-integration object
        self.vehicle_pims[imu.vehicle].integrateMeasurement(
            imu.accel, imu.gyro, self.imu_dt
        )

    def _create_imu_edge(self, vehicle: str):
        """
        Create IMU pre-integration factors for the associated vehicle and add it to the graph.
        """

        pim = self.vehicle_pims[vehicle]

        # Return if nothing to do
        if pim.deltaTij() == 0:
            return

        # Add to FG
        self.graph.push_back(
            gtsam.CombinedImuFactor(
                self.vehicle_pose_ids[vehicle][-1],
                self.vehicle_velocity_ids[vehicle][-1],
                self.next_id,
                self.next_id + 1,
                self.vehicle_bias_ids[vehicle][-1],
                self.next_id + 2,
                pim
            )
        )

        # Add estimated pose
        translation = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).translation()
        rotation = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).rotation()
        R = rotation.matrix()
        translation = translation + R @ pim.deltaPij()
        rotation = rotation.compose(pim.deltaRij())
        self.current_estimates.insert(
            self.next_id,
            gtsam.Pose3(rotation, translation)
        )

        # Add estimated velocity
        velocity = self.current_estimates.atVector(self.vehicle_velocity_ids[vehicle][-1])
        velocity = velocity + R @ pim.deltaVij()
        self.current_estimates.insert(
            self.next_id + 1,
            velocity
        )

        # Add zero bias
        self.current_estimates.insert(
            self.next_id + 2,
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        )

        self.vehicle_pose_ids[vehicle].append(self.next_id)
        self.vehicle_velocity_ids[vehicle].append(self.next_id + 1)
        self.vehicle_bias_ids[vehicle].append(self.next_id + 2)
        self.next_id += 3

        self.graph_outdated = True
        pim.resetIntegration()

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

    prior_noise = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    imu_noise = np.array([[0.1], [0.1], [0.1]])
    dt = 0.1

    priors = [
        Prior("A", np.array([[0], [0], [0], [0], [0]]), prior_noise),
        Prior("B", np.array([[0], [5], [np.pi / 4], [0], [0]]), prior_noise)
    ]
    imu_a = IMU("A", np.array([[0.1], [-0.1], [-0.01]]))
    imu_b = IMU("B", np.array([[0.3], [-0.3], [0.01]]))

    backend = Backend(priors, imu_noise, dt)

    for _ in range(100):
        backend.add_imu(imu_a)
        backend.add_imu(imu_b)
    backend._create_imu_edge("A")
    backend._create_imu_edge("B")
    backend._update()
    for _ in range(100):
        backend.add_imu(imu_a)
        backend.add_imu(imu_b)
    backend._create_imu_edge("A")
    backend._create_imu_edge("B")

    poses_1, covariances_1 = backend.get_full_trajectory("A")
    poses_2, covariances_2 = backend.get_full_trajectory("B")

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


