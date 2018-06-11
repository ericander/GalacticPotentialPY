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
    try:
        argmax = np.argwhere(rs == max(rs[postenc]))
    except IndexError:
        print('Error in data. Will flag run as terminated.')
        term = open(datadir + '../TERMINATED.out', 'w')
        term.close()
        return 'failed', 0, 0, 0, 0, 0

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

def findRUN(filename, rp = '', Etot = '', fret = '', fstr = '', funb = '',
        fcap = '', fMGC1 = '', term = '', strip = '', col = '',
        return_as_list = False):
    """ Function that finds an encounter number given a value of a specific
    parameter. Function works in two ways:
     (1) If parameter is given as tuple the function returns all runs within
         the specified range.
     (2) If parameter is given as a single value it find the encounter that
         has the closest to this value.
    If given multiple paramter the function tries to find the most suitable
    candidate.

    Positional Arguments:
        filename
            Name and path to output file in which function searches.

    Keyword Arguments:
        rp
            Pericentre distance
        Etot
            Total specific orbital energy
        fret
            Fraction of retained clusters.
        fstr
            Fraction of stripped clusters.
        funb
            Fraction of unbound clusters.
        fcap
            Fraction of captured clusters.
        fMGC1
            Fraction of MGC1-like clusters.
        term
            Terminated run
        strip
            Runs in which dwarf was stripped.
        col
            Runs in which dwarf collided with M31.
        return_as_list
            Returns runs in list
    """
    # Eric Andersson, 23-03-18.
    import numpy as np
    import pandas as pd
    K = 1/(0.001022**2)

    # Read in data.
    gc = pd.read_csv(filename,
            delimiter="\t", skiprows = 27,
            names = ["run", "rp", "ra", "e", "a", "v_inc", "Etot",
            "ntot", "nret", "fret", "nstr", "fstr", "nunb", "funb",
            "ncap", "fcap", "nMGC1", "fMGC1", "strip", "col",
            "rp_IC", "v_IC", "term"])
    run = np.array(gc.run)
    rpin = np.array(gc.rp)
    Etotin = np.array(gc.Etot)*K
    fretin = np.array(gc.fret)
    fstrin = np.array(gc.fstr)
    funbin = np.array(gc.funb)
    fcapin = np.array(gc.fcap)
    fMGC1in = np.array(gc.fMGC1)
    stripin = np.array(gc.strip)
    colin = np.array(gc.col)
    termin = np.array(gc.term)

    # Function for finding nearest.
    def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return idx

    # Set up masks if given ranges.
    mask = np.ones(rpin.size, dtype = bool)
    runs = []

    # Pericentre
    if not type(rp) == str:
        if type(rp) == tuple:
            mask = mask & (rp[0] < rpin) & (rp[1] > rpin)

    # Total specific energy.
    if not type(Etot) == str:
        if type(Etot) == tuple:
            mask = mask & (Etot[0] < Etotin) & (Etot[1] > Etotin)

    # Fraction of retained clusters.
    if not type(fret) == str:
        if type(fret) == tuple:
            mask = mask & (fret[0] < fretin) & (fret[1] > fretin)

    # Fraction of stripped clusters.
    if not type(fstr) == str:
        if type(fstr) == tuple:
            mask = mask & (fstr[0] < fstrin) & (fstr[1] > fstrin)

    # Fraction of unbound clusters.
    if not type(funb) == str:
        if type(funb) == tuple:
            mask = mask & (funb[0] < funbin) & (funb[1] > funbin)

    # Fraction of unbound clusters.
    if not type(fcap) == str:
        if type(fcap) == tuple:
            mask = mask & (fcap[0] < fcapin) & (fcap[1] > fcapin)

    # Fraction of MGC1-like clusters.
    if not type(fMGC1) == str:
        if type(fMGC1) == tuple:
            mask = mask & (fMGC1[0] < fMGC1in) & (fMGC1[1] > fMGC1in)

    # Stipped runs.
    if not type(strip) == str:
        mask = mask & (stripin == 1)

    # Collision runs.
    if not type(col) == str:
        mask = mask & (colin == 1)

    # Terminated runs.
    if not type(term) == str:
        mask = mask & (termin == 1)

    # Find specific encounters.
    if type(rp) == int:
        rp = float(rp)
    if type(Etot) == int:
        Etot = float(Etot)
    if type(fret) == int:
        fret = float(fret)
    if type(fstr) == int:
        fstr = float(fstr)
    if type(funb) == int:
        funb = float(funb)
    if type(fcap) == int:
        fcap = float(fcap)
    if type(fMGC1) == int:
        fMGC1 = float(fMGC1)

    # Pericentre
    if not type(rp) == str:
        if type(rp) == float:
            runs.append(find_nearest(rpin[mask], rp))

    # Total specific energy.
    if not type(Etot) == str:
        if type(Etot) == float:
            runs.append(find_nearest(Etotin[mask], Etot))

    # Fraction of retained clusters.
    if not type(fret) == str:
        if type(fret) == float:
            runs.append(find_nearest(fretin[mask], fret))

    # Fraction of stripped clusters.
    if not type(fstr) == str:
        if type(fstr) == float:
           runs.append(find_nearest(fstrin[mask], fstr))

    # Fraction of unbound clusters.
    if not type(funb) == str:
        if type(funb) == float:
           runs.append(find_nearest(funbin[mask], funb))

    # Fraction of unbound clusters.
    if not type(fcap) == str:
        if type(fcap) == float:
           runs.append(find_nearest(fcapin[mask], fcap))

    # Fraction of MGC1-like clusters.
    if not type(fMGC1) == str:
        if type(fMGC1) == float:
           runs.append(find_nearest(fMGC1in[mask], fMGC1))

    # Print resutls.
    print('Encounters within given ranges')
    if all(mask):
        print('No ranges was given or all encounters are contained within'\
                'given range!')
    else:
        masked_runs = run[mask]
        for r in masked_runs:
            print('RUN{0:03d}'.format(int(r)))

    print('\n\n Specific runs')
    if len(runs) == 0:
        print('No specific runs requested.')
    else:
        for r in runs:
            print('RUN{0:03d}'.format(int(run[r])))

    # Return
    if not return_as_list:
        return
    else:
        return runs, mask













