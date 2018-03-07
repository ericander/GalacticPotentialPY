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
            Separation between clusters and M31 at specific time.
        rs
            Separation between dwarf galaxy and M31 galaxy at specific
            time.
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

def MGC1_like(particles, run, t = '',
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
    if type(rs) == str or type(t) == str:
        (t, xs, ys, zs, vx, vy, vz, _, _, _) = read.satellite(datadir)
        rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # Index where encounter occurs.
    t0i = np.argwhere(t==min(t[(t>0)]))[0][0]

    # Filter out data before encounter.
    postenc = list(range(t0i, r[0].size))

    # Argument at maximum after encounter.
    argmax = np.argwhere(rs == max(rs[postenc]))

    # Retained clusters.
    if type(ret) == str:
        ret, nret = retained(particles, r[...,argmax], rs[argmax], run,
                datadir = datadir + './../../')

    # Unbound clusters.
    if type(unb) == str:
        unb, nunb = unbound(particles, Ek[:,-1], Ep[:,-1], ret = ret,
                datadir = datadir + './../../')

    # Remove retained and unbound clusters.
    M31GC = (ret == False) & (unb == False)
    bound_particles = np.array(particles)[M31GC]

    # Find MGC1-like clusters
    MGC1 = np.zeros([npar], dtype=bool)
    n = 0

    # Needed rounding function.
    def myround(x, base = 100):
        for i in range(x.size):
            x[i] = int(base * round(float(x[i])/base))
        return x

    # Check which clusters pass 200 kpc at least 2 times.
    for par in bound_particles:
        rr = np.round(r[par][postenc], 0)
        arg = np.argwhere(rr == 200)
        # Ensure that we dont double-count due to unprecise rounding.
        arg = list(set(myround(arg[...,0])))
        if len(arg) > 1:
            n+=1
            MGC1[par] = True

    return MGC1, n, unb, nunb, ret, nret
