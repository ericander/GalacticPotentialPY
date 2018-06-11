#! /usr/bin/env python3
#=======================================================================
# draw.py
#
# Facilities for drawing initial parameters for dwarf trajectory.
#
# Eric Andersson, 2018-01-15
#=======================================================================


def incoming_angle(lim):
    """Draws incoming angle for dwarf galaxies coming in uniformly from
    a any direction.

    Positional Arguments:
        lim
            Range for incoming angle.
    """
    # Eric Andersson, 2018-01-15
    import numpy as np

    theta = np.arcsin(1 - np.random.uniform(
            np.sin(lim[0]), np.sin(lim[1]), 1)[0])

    return theta

def impact_angle(lim):
    """Draws impact angle for dwarf galaxy.

    Positional Arguments:
        lim
            Range for impact angle.
    """
    # Eric Andersson, 2018-01-15
    import numpy as np

    phi = np.random.uniform(lim[0], lim[1], 1)[0]

    return phi

def impact_parameter(v, R_max, b_min):
    """Draws impact parameter for dwarf galaxy.

    Positional Arguments:
        v
            Incoming velocity of dwarf galaxy.
        R_max
            Maximum pericentre distance.
        b_min
            Minimum impact paramter.
    """
    # Eric Andersson, 2018-01-15
    import numpy as np
    from . import constants
    from . import read

    # Constants.
    G = constants.gravitational_constant(unit = 'kpc(km/s)^2Msun')
    (M_M31, ) = read.setup(param = ['M_M31'], datadir = './')

    # Escape speed for particle at R_max.
    v_esc = np.sqrt(2 * G * M_M31 / R_max)

    # Maximum impact paramter.
    b_max =  np.sqrt(1 + v_esc**2 / v**2) * R_max

    b = np.sqrt(np.random.uniform(b_min**2, b_max**2, 1)[0])

    return b


def incoming_velocity(trajectory, r, v_max = 29979.2):
    """Draws incoming velocity for dwarf galaxy.

    Positional Arguments:
        trajectory
            Type of trajectory.
        r
            Distance between M31 and dwarf galaxy.

    Keyword Arguments:
        v_max
            Maximum incomming velocity.
    """
    # Eric Andersson, 2018-01-15
    import numpy as np
    from . import constants
    from . import read

    # Constants
    G = constants.gravitational_constant(unit = 'kpc(km/s)^2Msun')
    (M_M31, ) = read.setup(param = ['M_M31'], datadir = './')

    # Velocity of parabolic trajectory
    vp = np.sqrt(2 * G * M_M31 / r)

    if trajectory == 'parabolic':
        return vp
    elif trajectory == 'bound':
        return vp * np.random.uniform(0, 1, 1)[0]
    elif trajectory == 'hyperbolic':
        return vp * np.random.uniform(1, v_max, 1)[0]
    else:
        raise ValueError(
                trajectory + ' is not a valid trajectory')

def pericentre(N, lim, rdist = 'rsquared'):
    """Generates a pericentre position for a dwarf galaxy.

    Positional Arguments:
        N
            Number of pericentres.
        lim
            Limits of pericentre length.

    """
    # Eric Andersson, 2018-02-14.
    from . import compute

    return compute.sample_sphere(lim, N, dist = 'random',
            rdist = rdist)[0]

def energy(N, lim):
    """Draws a random total energy in the range given by lim.

    Positional Arguments:
        N
            Number of energies.
        lim
            Limits of energy.

    """
    # Eric Andersson, 14-02-2018.
    import numpy as np

    return np.random.uniform(lim[0], lim[1], N)

def angular_momentum_direction(N):
    """Generates a normalized direction of angular momentum.

    Keyword Arguments:
        N
            Number of angular momentum directions.
    """
    # Eric Andersson, 14-02-2018
    from . import compute

    return compute.sample_sphere((0, 1), N, dist = 'random')[0]


