#! /usr/bin/env python3
#=======================================================================
# read.py
#
# Facilities for reading the data from Galactic potential particle
# integrator.
#
# Eric Andersson, 2018-01-11
#=======================================================================

def info(run, info, datadir = './../'):
    """Reads in data about a simulated encounters.

    Positional Arguments:
        run
            The encounter which is investigated. run = 5 -> RUN005
        info
            Information that is wanted.

    Keyword Arguments:
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-11
    if not 'RUN {}'.format(run) in open(datadir + 'info.txt').read():
        raise ValueError('RUN{0:03d}'.format(run) + ' does not exist')

    # Find line where encounter info is stored.
    read = False
    n = 0
    filename = datadir + 'info.txt'
    substr = 'RUN {}'.format(run)
    for line in open(filename):
        for subline in line.strip().split('\n'):
            if substr in line:
                read = True
            if read:
                n+=1
                for i in range(len(info)):
                    words = line.split()
                    if not len(words) == 0:
                        if words[0] == info[i]:
                            try:
                                info[i] = float(words[2])
                            except ValueError:
                                info[i] = words[2]
            if n < 27:
                break

    return tuple(info)

    for line in open(datadir + 'info.txt'):
        words = line.split()
        if not len(words) == 0:
            if words[0] == info:
                try:
                    return float(words[2])
                except ValueError:
                    return words[2]

    raise ValueError(info + " is not an available parameter.")

def setup(param, datadir = './'):
    """Reads the parameters set for the encounter. The script will
    search for a file setup.txt

    Positional Arguments:
        param
            List of paramters needed from setup.txt

    Keyword Arguments:
        datadir
            Directory of setup.txt
    """
    # Eric Andersson, 2018-01-15

    # Search for parameters.
    for i in range(len(param)):
        for line in open(datadir + 'setup.txt'):
            words = line.split()
            if not len(words) == 0:
                if words[0] == param[i]:
                    try:
                        param[i] = float(words[2])
                    except ValueError:
                        param[i] = words[2]

    return tuple(param)

def encounter(param, datadir = './'):
    """Reads information about encounters in a simulation set.

    Positional Arguments:
        param
            Paramter that will be searched for.

    Keyword Arguments:
        datadir
            Directory of encounter information
    """
    # Eric Andersson, 2018-01-17
    import numpy as np

    # Search for parameter
    data = []
    for line in open(datadir + 'Encounters.txt'):
        words = line.split()
        if not len(words) == 0:
            if words[0] == param:
                try:
                    data.append(float(words[2]))
                except ValueError:
                    data.append(words[2])

    return np.array(data)

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
    K = 1# constants.conversion('kpc/Myr->km/s')

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
    Ek = np.zeros([npar, ndat])
    Ep = np.zeros([npar, ndat])

    # Read data.
    for par in particles:
        data = pd.read_csv(datadir + "particle_{}.txt".format(par),
                delimiter="\t",
                names = ["t", "x", "y", "z", "vx", "vy", "vz",
                    "Ek", "Ep"])
        # Look for bugg in data.
        try:
            t[par] = data.t
        except ValueError:
            error = open(datadir + '../TERMINATED.out', 'w')
            error.close()
            continue
        t[par] = data.t
        x[par] = data.x
        y[par] = data.y
        z[par] = data.z
        vx[par] = data.vx * K
        vy[par] = data.vy * K
        vz[par] = data.vz * K
        Ek[par] = data.Ek
        Ep[par] = data.Ep

    return (t, x, y, z, vx, vy, vz, Ek, Ep)


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
    K = 1# constants.conversion('kpc/Myr->km/s')

    data = pd.read_csv(datadir + "satellite.txt",
            delimiter="\t",
            names = ["t", "x", "y", "z", "vx", "vy", "vz", "col",
                "Ek", "Ep"])

    t = np.array(data.t)
    x =  np.array(data.x)
    y = np.array(data.y)
    z = np.array(data.z)
    vx = np.array(data.vx) * K
    vy = np.array(data.vy) * K
    vz = np.array(data.vz) * K
    col = np.array(data.col)
    Ek = np.array(data.Ek)
    Ep = np.array(data.Ep)
    return (t, x, y, z, vx, vy, vz, col, Ek, Ep)

def GC_sample(datadir = './'):
    """Reads the generated sample of clusters.

    Keyword Arguments:
        datadir
            Directory of sample
    """
    # Eric Andersson, 2018-01-15
    import numpy as np
    import pandas as pd

    # Read data.
    sample = np.loadtxt(datadir + 'GC_sample.txt')

    return sample
