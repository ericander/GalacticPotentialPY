#! /usr/bin/env python3
#=======================================================================
# analyse.py
#
# Facilities for analyzing the data from Galactic potential N-body
# integrator.
#
# Eric Andersson, 2018-01-12
#=======================================================================

def retained(particles, r, rs, datadir = './data'):
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

    Keyword Arguments:
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-12
    import numpy as np
    from . import read
    from . import compute
    from . import constants

    # Encounter parameters
    Ms = read.info('M_s')
    M_M31 = constants.M31_mass()

    # Compute roche-lobe radius for dwarf.
    rrl = compute.roche_lobe(rs, Ms, M_M31)

    # Check whether particles are within dwarf Roche-lobe
    n = 0
    ret = np.zeros(len(particles))
    for par in range(len(particles)):
        if rrl > abs(rs - r[par]):
            n+=1
            ret[par] = 1

    return (ret == 1), n

def unbound(particles, r, v, ret = '', rs = '', datadir = './data'):
    """Counts number of unbound clusters.

    Positional Arguments:
        particles
            List of all particles.
        r
            Separation between clusters and M31 at end of simulation.
        v
            Velocity of all clusters.

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
    if type(ret) == type(''):
        if rs == '':
            raise ValueError("Did not provide rs")
        (ret, _) = retained(particles, r, rs, datadir)

    # Compute potential energy of particles.
    M_M31 = constants.M31_mass()
    Ep = compute.pointmass_potential(r, M_M31)

    # Compute kinetic energy of particles.
    Ek = 0.5 * v**2

    # Check whether bound or not.
    n = 0
    unbound = np.zeros(npar)
    for par in range(npar):
        if Ep[par] < Ek[par]:
            if not ret[par]:
                unbound[par] = 1
                n+=1

    return (unbound == 1), n

def MGC1_like(particles,
        rs = np.zeros(0), r = np.zeros(0), v = np.zeros(0),
        datadir = './data/'):
    """Computes the fraction of MGC1-like clusters in an encounter.

    Positional Arguments:
        particles
            List of all particles.

    Keyword Arguments:
        r
            Distance from M31 for all clusters.
        v
            Velocity of all clusters
        rs
            Distance from M31 to the dwarf galaxy.
        datadir
            Directory of data.
    """
    # Eric Andersson, 2018-01-12
    import numpy as np
    from . import read

    npar = len(particles)

    # Load data unless provided
    if r.size == 0 or v.size == 0:
        (_, x, y, z, vx, vy, vz) = read.particle(particles, datadir)
        r = np.zeros(x.shape)
        v = np.zeros(x.shape)
        r[:] = np.sqrt(x[:]**2 + y[:]**2 + z[:]**2)
        v[:] = np.sqrt(vx[:]**2 + vy[:]**2 + vz[:]**2)
    if rs.size == 0:
        (_, xs, ys, zs, vx, vy, vz, _) = read.satellite(datadir)
        rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # Index at closest approach
    xpi = np.argmin(rs)

    # Store only data at last step
    rl = np.zeros(npar)
    vl = np.zeros(npar)
    for par in range(npar):
        rl[par] = r[par][-1]
        vl[par] = v[par][-1]

    # Retained clusters.
    ret, nret = retained(particles, rl, rs[-1], datadir)

    # Unbound clusters.
    unb, nunb = unbound(particles, rl, vl, ret = ret, datadir = datadir)

    # Determine which or remaining clusters are MGC1 like.
    raise NotImplementedError("Function is not implemented yet.")
