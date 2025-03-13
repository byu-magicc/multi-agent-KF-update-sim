from functools import partial
from typing import List, Optional

import gtsam
from gtsam.symbol_shorthand import X
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


# Noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
GLOBAL_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, np.inf]))
RANGE_NOISE = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)


def error_global(measurement: np.ndarray,
                 this: gtsam.CustomFactor,
                 values: gtsam.Values,
                 jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Global Factor error function
    :param measurement: Global measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    key = this.keys()[0]
    estimate = values.atPose2(key)
    estimate = np.array([estimate.x(), estimate.y(), estimate.theta()])
    error = estimate - measurement

    if jacobians is not None:
        jacobians[0] = np.eye(3)

    return error


def main():
    # Create empty graph
    graph = gtsam.NonlinearFactorGraph()

    # Create keys corresponding to each pose for two vehicles
    X1 = X(1)
    X2 = X(2)
    X3 = X(3)
    X4 = X(4)
    X5 = X(5)

    X6 = X(6)
    X7 = X(7)
    X8 = X(8)
    X9 = X(9)
    X10 = X(10)

    # Add the priors
    graph.add(
        gtsam.PriorFactorPose2(X1, gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE)
    )
    graph.add(
        gtsam.PriorFactorPose2(X6, gtsam.Pose2(0.0, 5.0, 0.0), PRIOR_NOISE)
    )

    # Add the odometry factors
    graph.add(
        gtsam.BetweenFactorPose2(X1, X2, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )
    graph.add(
        gtsam.BetweenFactorPose2(X2, X3, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )
    graph.add(
        gtsam.BetweenFactorPose2(X3, X4, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )
    graph.add(
        gtsam.BetweenFactorPose2(X4, X5, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )

    graph.add(
        gtsam.BetweenFactorPose2(X6, X7, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )
    graph.add(
        gtsam.BetweenFactorPose2(X7, X8, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )
    graph.add(
        gtsam.BetweenFactorPose2(X8, X9, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )
    graph.add(
        gtsam.BetweenFactorPose2(X9, X10, gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE)
    )

    # Add range factor
    graph.add(
        gtsam.RangeFactorPose2(X3, X8, 5.0, RANGE_NOISE)
    )

    # Add global measurement
    graph.add(
        gtsam.CustomFactor(GLOBAL_NOISE, [X5], partial(error_global, np.array([8.0, 1.0, 0.0])))
    )

    # Print graph
    print("Factor Graph:\n{}".format(graph))

    initial_estimate = gtsam.Values()
    initial_estimate.insert(X1, gtsam.Pose2(-0.25, 0.20, 0.15))
    initial_estimate.insert(X2, gtsam.Pose2(2.30, 0.10, -0.20))
    initial_estimate.insert(X3, gtsam.Pose2(4.10, 0.10, 0.10))
    initial_estimate.insert(X4, gtsam.Pose2(6.10, 0.20, -0.15))
    initial_estimate.insert(X5, gtsam.Pose2(8.10, 0.10, 0.20))

    initial_estimate.insert(X6, gtsam.Pose2(0.20, 5.10, -0.10))
    initial_estimate.insert(X7, gtsam.Pose2(2.10, 5.00, 0.20))
    initial_estimate.insert(X8, gtsam.Pose2(4.10, 5.10, -0.20))
    initial_estimate.insert(X9, gtsam.Pose2(6.00, 5.10, 0.10))
    initial_estimate.insert(X10, gtsam.Pose2(8.10, 5.10, -0.10))

    # Solve with Leveberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    # Calculate marginal covariances for all poses
    marginals = gtsam.Marginals(graph, result)
    covariances = np.array([marginals.marginalCovariance(X(i)) for i in range(1, 11)])
    print(covariances)

    # Plot poses
    plt.figure()
    poses = np.array([
        [result.atPose2(X(i)).x(), result.atPose2(X(i)).y(), result.atPose2(X(i)).theta()]
        for i in range(1, 11)
    ]).T
    print(poses)
    plt.plot(poses[0, :5], poses[1, :5], color="b", marker="o", label="Vehicle 1")
    plt.plot(poses[0, 5:], poses[1, 5:], color="g", marker="o", label="Vehicle 2")

    # Plot covariance ellipses
    for i in range(10):
        num_sigma = 2
        cov = covariances[i]
        mean = poses[:2, i]
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
    plt.show()


if __name__ == "__main__":
    main()
