import numpy as np


def line_trajectory(n_points, start_location, end_location):
    """
    Generates a straight line trajectory. Poses are returned in inertial frame.

    Parameters:
    n_points (int): The number of points to generate.
    start_location (np.array): The starting location of the line. [x, y]
    end_location (np.array): The ending location of the line. [x, y]

    Returns:
    np.array: The position and heading of the robot along a line trajectory. [[x1, y1, psi1].T, ... ].T
    """
    position = np.array([start_location + i * (end_location - start_location) / (n_points - 1)
                         for i in range(n_points)])
    heading = np.arctan2(end_location[1] - start_location[1],
                         end_location[0] - start_location[0]) * np.ones(n_points)
    return np.column_stack([position, heading]).T


def arc_trajectory(n_points, start_location, end_location, angle):
    """
    Generates an arc trajectory.

    Parameters:
    n_points (int): The number of points to generate.
    start_location (np.array): The starting location of the arc. [x, y]
    end_location (np.array): The ending location of the arc. [x, y]
    angle (float): The angle of the arc in degrees at the starting point, measured from the 
        line between the start and end points.

    Returns:
    np.array: The position and heading of the robot along an arc trajectory. [[x1, y1, psi1].T, ... ].T
    """
    angle = np.deg2rad(angle)

    # If the angle is 0, generate a line trajectory to avoid numerical instability
    if angle == 0:
        return line_trajectory(n_points, start_location, end_location)

    x = np.linalg.norm(end_location - start_location) / 2
    y = x * np.tan(np.abs(angle) - np.pi/2)
    center = np.array([x, y])
    radius = np.linalg.norm(center)

    start_angle = np.pi/2 + np.abs(angle)
    end_angle = np.pi/2 - np.abs(angle)
    delta_angle = (end_angle - start_angle) / (n_points - 1)

    points = np.array([center + radius * np.array([np.cos(start_angle + i * delta_angle),
                                                   np.sin(start_angle + i * delta_angle)])
                       for i in range(n_points)])
    heading = np.array([np.abs(angle) + i * delta_angle for i in range(n_points)])

    # Flip, rotate, and translate the points to the correct location
    points[:, 1] *= np.sign(angle)
    heading *= np.sign(angle)
    rotation_angle = np.arctan2(end_location[1] - start_location[1],
                                end_location[0] - start_location[0])
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    points = np.dot(points, rotation_matrix.T) + start_location
    heading += rotation_angle

    return np.column_stack([points, heading]).T


def sine_trajectory(n_points, start_location, end_location, amplitude, n_periods):
    """
    Generates a sine trajectory.

    Parameters:
    n_points (int): The number of points to generate.
    start_location (np.array): The starting location of the sine wave. [x, y]
    end_location (np.array): The ending location of the sine wave. [x, y]
    amplitude (float): The amplitude of the sine wave.
    n_periods (int): The number of periods in the sine wave. Recommended to be an integer,
        or an integer plus 0.5.

    Returns:
    np.array: The position and heading of the robot along a sine trajectory. [[x1, y1, psi1].T, ... ].T
    """

    # Generate the sine wave
    length = np.linalg.norm(end_location - start_location)
    x = np.linspace(0, n_periods, n_points) * length / n_periods
    y = amplitude * np.sin(2*np.pi * x * n_periods / length)
    y_prime = amplitude * 2*np.pi * n_periods / length * np.cos(2*np.pi * x * n_periods / length)
    heading = np.arctan(y_prime)

    # Rotate and translate the sine wave to the correct location
    rotation_angle = np.arctan2(end_location[1] - start_location[1],
                                end_location[0] - start_location[0])
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    points = np.dot(np.column_stack([x, y]), rotation_matrix.T) + start_location
    heading += rotation_angle

    return np.column_stack([points, heading]).T
