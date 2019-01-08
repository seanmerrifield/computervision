from robot_class import robot
from math import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------
# this helper function displays the world that a robot is in
# it assumes the world is a square grid of some given size
# and that landmarks is a list of landmark positions(an optional argument)
def display_world(world_size, position, landmarks=None):
    # using seaborn, set background grid to gray
    sns.set_style("dark")

    # Plot grid of values
    world_grid = np.zeros((world_size + 1, world_size + 1))

    # Set minor axes in between the labels
    ax = plt.gca()
    cols = world_size + 1
    rows = world_size + 1

    ax.set_xticks([x for x in range(1, cols)], minor=True)
    ax.set_yticks([y for y in range(1, rows)], minor=True)

    # Plot grid on minor axes in gray (width = 1)
    plt.grid(which='minor', ls='-', lw=1, color='white')

    # Plot grid on major axes in larger width
    plt.grid(which='major', ls='-', lw=2, color='white')

    # Create an 'o' character that represents the robot
    # ha = horizontal alignment, va = vertical
    ax.text(position[0], position[1], 'o', ha='center', va='center', color='r', fontsize=30)

    # Draw landmarks if they exists
    if (landmarks is not None):
        # loop through all path indices and draw a dot (unless it's at the car's location)
        for pos in landmarks:
            if (pos != position):
                ax.text(pos[0], pos[1], 'x', ha='center', va='center', color='purple', fontsize=20)

    # Display final result
    plt.show()


# --------
# this routine makes the robot data
# the data is a list of measurements and movements: [measurements, [dx, dy]]
# collected over a specified number of time steps, N
#
def make_data(N, num_landmarks, world_size, measurement_range, motion_noise,
              measurement_noise, distance):
    # check that data has been made
    try:
        check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise)
    except ValueError:
        print('Error: You must implement the sense function in robot_class.py.')
        return []

    complete = False

    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
    r.make_landmarks(num_landmarks)

    while not complete:

        data = []

        seen = [False for row in range(num_landmarks)]

        # guess an initial motion
        orientation = random.random() * 2.0 * pi
        dx = cos(orientation) * distance
        dy = sin(orientation) * distance

        for k in range(N - 1):

            # collect sensor measurements in a list, Z
            Z = r.sense()

            # check off all landmarks that were observed
            for i in range(len(Z)):
                seen[Z[i][0]] = True

            # move
            while not r.move(dx, dy):
                # if we'd be leaving the robot world, pick instead a new direction
                orientation = random.random() * 2.0 * pi
                dx = cos(orientation) * distance
                dy = sin(orientation) * distance

            # collect/memorize all sensor and motion data
            data.append([Z, [dx, dy]])

        # we are done when all landmarks were observed; otherwise re-run
        complete = (sum(seen) == num_landmarks)

    print(' ')
    print('Landmarks: ', r.landmarks)
    print(r)

    return data


def check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise):
    # make robot and landmarks
    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
    r.make_landmarks(num_landmarks)

    # check that sense has been implemented/data has been made
    test_Z = r.sense()
    if (test_Z is None):
        raise ValueError



def initialize_constraints(N, num_landmarks, world_size):
    ''' This function takes in a number of time steps N, number of landmarks, and a world_size,
        and returns initialized constraint matrices, omega and xi.'''

    ## Recommended: Define and store the size (rows/cols) of the constraint matrix in a variable
    size = 2 * int(N) + 2 * int(num_landmarks)
    x0 = 0
    y0 = int(N)
    Lx0 = 2 * int(N)
    Ly0 = 2 * int(N) + int(num_landmarks)

    ## TODO: Define the constraint matrix, Omega, with two initial "strength" values
    ## for the initial x, y location of our robot
    omega = np.zeros([size, size])
    omega[x0][x0] = 1
    omega[y0][y0] = 1

    ## TODO: Define the constraint *vector*, xi
    ## you can assume that the robot starts out in the middle of the world with 100% confidence
    xi = np.zeros([size,1])
    xi[x0] = world_size / 2
    xi[y0] = world_size / 2
    return omega, xi


def update_measurement(omega, Xi, xi, Li, distance, noise):
    """
    Updates constraint matrix omega and position xi based on measurements to landmark locations
    omega:     Constraint matrix
    Xi:        Position vector
    xi:        Position index at time i
    Li:        Landmark location index
    distance:  Distance measured between landmark and robot at time i

    return: omega, Xi

    """

    # Update constraint between xi and Li

    #xi - Li = -distance
    omega[xi][xi] += 1.0/noise    #xi
    omega[xi][Li] -= 1.0/noise    #Li
    Xi[xi] -= distance/noise

    #Li - xi = distance
    omega[Li][Li] += 1.0/noise   #Li
    omega[Li][xi] -= 1.0/noise    #xi
    Xi[Li] += distance/noise

    # xi - Li = -distance
    # omega[xi][xi] += 1.0  # xi
    # omega[xi][Li] -= 1.0  # Li
    # Xi[xi] -= distance
    #
    # # Li - xi = distance
    # omega[Li][Li] += 1.0  # Li
    # omega[Li][xi] -= 1.0  # xi
    # Xi[Li] += distance

    return omega, Xi


def update_position(omega, Xi, xi, movement, noise):
    """
    Updates constraint matrix omega and position xi based on robot movement
    omega:     Constraint matrix
    Xi:        Position vector
    xi:        Position index at time i
    moement:   Movement of robot at time i

    return: omega, Xi

    """
    # xi-1 - xi = -dx
    omega[xi-1][xi-1] += 1.0/noise  #xi-1
    omega[xi-1][xi] -= 1.0/noise  #xi
    Xi[xi-1] -= movement/noise

    # xi - xi-1 = dx
    omega[xi][xi] += 1.0/noise  #xi
    omega[xi][xi-1] -= 1.0/noise #xi-1
    Xi[xi] += movement/noise

    # # xi-1 - xi = -dx
    # omega[xi - 1][xi - 1] += 1.0  # xi-1
    # omega[xi - 1][xi] -= 1.0  # xi
    # Xi[xi - 1] -= movement
    #
    # # xi - xi-1 = dx
    # omega[xi][xi] += 1.0  # xi
    # omega[xi][xi - 1] -= 1.0  # xi-1
    # Xi[xi] += movement

    return omega, Xi


## TODO: Complete the code to implement SLAM

## slam takes in 6 arguments and returns mu,
## mu is the entire path traversed by a robot (all x,y poses) *and* all landmarks locations
def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    ## TODO: Use your initilization to create constraint matrices, omega and xi
    omega, Xi = initialize_constraints(N, num_landmarks, world_size)

    # Setup indexes
    x0 = 0
    y0 = int(N)
    Lx0 = 2 * int(N)
    Ly0 = 2 * int(N) + num_landmarks

    ## TODO: Iterate through each time step in the data
    ## get all the motion and measurement data as you iterate
    for i, (measurements, motions) in enumerate(data):
        xi = x0 + (i+1)
        yi = y0 + (i+1)

        #         #Measurements and motions at a given position xi
        #         measurements, motions = data[time_step]

        ## TODO: update the constraint matrix/vector to account for all *measurements*
        ## this should be a series of additions that take into account the measurement noise
        for loc_i, dx, dy in measurements:
            Lxi = Lx0 + loc_i
            Lyi = Ly0 + loc_i
            omega, Xi = update_measurement(omega, Xi, xi, Lxi, dx, measurement_noise)
            omega, Xi = update_measurement(omega, Xi, yi, Lyi, dy, measurement_noise)

        ## TODO: update the constraint matrix/vector to account for all *motion* and motion noise
        omega, Xi = update_position(omega, Xi, xi, motions[0], motion_noise)
        omega, Xi = update_position(omega, Xi, yi, motions[1], motion_noise)

    ## TODO: After iterating through all the data
    ## Compute the best estimate of poses and landmark positions
    ## using the formula, omega_inverse * Xi
    mu = np.linalg.inv(np.matrix(omega))*Xi
    # mu = None
    return omega, Xi, mu  # return `mu`

# a helper function that creates a list of poses and of landmarks for ease of printing
# this only works for the suggested constraint architecture of interlaced x,y poses
def get_poses_landmarks(mu, N, num_landmarks):

    x0 = 0
    y0 = N
    Lx0 = 2*N
    Ly0 = 2*N + num_landmarks

    # create a list of poses
    poses = []
    for i in range(N):
        poses.append((mu[i].item(), mu[i+y0].item()))

    # create a list of landmarks
    landmarks = []
    for i in range(num_landmarks):
        landmarks.append((mu[Lx0+i].item(), mu[Ly0+i].item()))

    # return completed lists
    return poses, landmarks


def print_all(poses, landmarks):
    print('\n')
    print('Estimated Poses:')
    for i in range(len(poses)):
        print('['+', '.join('%.3f'%p for p in poses[i])+']')
    print('\n')
    print('Estimated Landmarks:')
    for i in range(len(landmarks)):
        print('['+', '.join('%.3f'%l for l in landmarks[i])+']')
