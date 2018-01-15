#! /usr/bin/env python3
#=======================================================================
# read.py
#
# Facilities for reading the data from Galactic potential particle
# integrator.
#
# Eric Andersson, 2018-01-11
#=======================================================================

def info(info, datadir = './../'):
    """Reads in data about a simulated set of encounters.

    Positional Arguments:
        info
            Information that is wanted.

    Keyword Arguments:
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-11
    lines = sum(1 for line in open(datadir + 'info.txt'))

    for line in open(datadir + 'info.txt'):
        words = line.split()
        if not len(words) == 0:
            if words[0] == info:
                return float(words[2])

    raise ValueError(info + " is not an available parameter.")

def particle(particles,
        datadir = './data/'):
    """Reads in data for particles.

    Positional Arguments:
        particles
            List of all particle numbers.

    Keyword Arguments:
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-11
    import numpy as np
    import pandas as pd
    from . import constants

    # Constants
    K = constants.conversion('kpc/Myr->km/s')

    # Count number of particles and number of data points.
    npar = len(particles)
    ndat = sum(1 for line in open(
        datadir + 'particle_{}.txt'.format(particles[0])))

    # Initiate arrays for holding data.
    t = np.zeros([npar, ndat])
    x = np.zeros([npar, ndat])
    y = np.zeros([npar, ndat])
    z = np.zeros([npar, ndat])
    vx = np.zeros([npar, ndat])
    vy = np.zeros([npar, ndat])
    vz = np.zeros([npar, ndat])

    # Read data.
    for par in particles:
        data = pd.read_csv(datadir + "particle_{}.txt".format(par),
                delimiter="\t",
                names = ["t", "x", "y", "z", "vx", "vy", "vz"])
        t[par] = data.t
        x[par] = data.x
        y[par] = data.y
        z[par] = data.z
        vx[par] = data.vx * K
        vy[par] = data.vy * K
        vz[par] = data.vz * K

    return (t, x, y, z, vx, vy, vz)


def satellite(datadir = './data/'):
    """Reads in position data of satellite.

    Keyword Arguments:
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-11
    import numpy as np
    import pandas as pd
    from . import constants

    # Constants
    K = constants.conversion('kpc/Myr->km/s')

    data = pd.read_csv(datadir + "satellite.txt",
            delimiter="\t",
            names = ["t", "x", "y", "z", "vx", "vy", "vz", "col"])

    t = np.array(data.t)
    x =  np.array(data.x)
    y = np.array(data.y)
    z = np.array(data.z)
    vx = np.array(data.vx) * K
    vy = np.array(data.vy) * K
    vz = np.array(data.vz) * K
    col = np.array(data.col)
    return (t, x, y, z, vx, vy, vz, col)
