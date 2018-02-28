#! /usr/bin/env python3
#=======================================================================
# analyse.py
#
# Facilities for analyzing the data from Galactic potential N-body
# integrator.
#
# Eric Andersson, 2018-01-12
#=======================================================================

def retained(particles, r, rs, run, Ms = 'Ms', MM31 = 'MM31',
        datadir = './../'):
    """Computes the number of cluster retained by the dwarf galaxy
    in the encounter.

    Positional Arguments:
        particles
            List of all particles.
        r
            Separation between clusters and M31 at end of simulation.
        rs
            Separation between dwarf galaxy and M31 at end of
            simulation.
        run
            Which encounter is investigated.

    Keyword Arguments:
        Ms
            Dwarf galaxy mass.
        MM31
            M31 mass.
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-12
    import numpy as np
    from . import read
    from . import compute
    from . import constants

    # Encounter parameters
    if type(Ms) == str:
        (Ms,) = read.info(run = run, info = ['M_s'], datadir = datadir)
    if type(MM31) == str:
        MM31 = constants.M31_mass()

    # Compute roche-lobe radius for dwarf.
    rrl = compute.roche_lobe(rs, Ms, MM31)

    # Check whether particles are within dwarf Roche-lobe
    n = 0
    ret = np.zeros(len(particles))
    for par in range(len(particles)):
        if rrl > abs(rs - r[par]):
            n+=1
            ret[par] = 1

    return (ret == 1), n

def unbound(particles, Ek, Ep, rs = '', r = '', ret = '',
        datadir = './data'):
    """Counts number of unbound clusters.

    Positional Arguments:
        particles
            List of all particles.
        Ek
            Kinetic energy of particles
        Ep
            Potential energy of particles

    Keyword Arguments:
        ret
            List of all retained clusters.
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-14
    from . import constants
    from . import compute
    import numpy as np

    npar = len(particles)

    # Count retained clusters unless provided.
    if type(ret) == str:
        if rs == '':
            raise ValueError("Did not provide rs")
        (ret, _) = retained(particles, r, rs, datadir)

    # Check whether bound or not.
    n = 0
    unbound = np.zeros(npar)
    for par in particles:
        if Ep[par] + Ek[par] > 0:
            if not ret[par]:
                unbound[par] = 1
                n+=1

    return (unbound == 1), n

def MGC1_like(particles, run,
        rs = '', r = '', v = '', ret = '', unb = '', Ek = '', Ep = '',
        datadir = './data/'):
    """Computes the fraction of MGC1-like clusters in an encounter.

    Positional Arguments:
        particles
            List of all particles.
        run
            Encounter number.

    Keyword Arguments:
        r
            Distance from M31 for all clusters.
        v
            Velocity of all clusters
        rs
            Distance from M31 to the dwarf galaxy.
        ret
            Retained clusters. Calculated unless provided.
        unb
            Unbound clusters. Calculated unless provided.
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-12
    import numpy as np
    from . import read

    npar = len(particles)

    # Load data unless provided
    if type(r) == str or type(v) == str or type(Ek) == str or type(Ep) == str:
        (_, x, y, z, vx, vy, vz, Ek, Ep) = read.particle(particles,
                datadir)
        r = np.zeros(x.shape)
        v = np.zeros(x.shape)
        r[:] = np.sqrt(x[:]**2 + y[:]**2 + z[:]**2)
        v[:] = np.sqrt(vx[:]**2 + vy[:]**2 + vz[:]**2)
    if type(rs) == str:
        (_, xs, ys, zs, vx, vy, vz, _, _, _) = read.satellite(datadir)
        rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # Index at closest approach
    xpi = np.argmin(rs)

    # Store only data at last step
    rl = r[:,-1]
    vl = v[:,-1]

    # Retained clusters.
    if type(ret) == str:
        ret, nret = retained(particles, rl, rs[-1], run,
                datadir = datadir + './../../')

    # Unbound clusters.
    if type(unb) == str:
        unb, nunb = unbound(particles, Ek[:,-1], Ep[:,-1], ret = ret,
                datadir = datadir + './../../')

    # Remove retained and unbound clusters.
    M31GC = (ret == False) & (unb == False)
    bound_particles = np.array(particles)[M31GC]

    # Find index at with dwarf crosses 300 kpc a second time.
    postenc = list(range(xpi, r[0].size))
    rsp = rs[postenc]
    failed = False
    try:
        rsp = min(rsp[rsp > 300])
        xai = np.where(rs == rsp)[0][0]
    except ValueError:
        failed = True

    # Find MGC1-like clusters
    MGC1 = np.zeros([npar], dtype=bool)
    n = 0
    if not failed:
        for par in bound_particles:
            mask = list(range(xai, r[par].size))
            if min(r[par][mask]) < 200:
                if max(r[par][mask]) > 200:
                    MGC1[par] = True
                    n+=1
    return MGC1, n, unb, ret
