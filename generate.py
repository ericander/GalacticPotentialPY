#! /usr/bin/env python3
#=======================================================================
# generate.py
#
# Facilities for generating data for Galactic potential N-body
# integrator.
#
# Eric Andersson, 2018-01-15
#=======================================================================

def GC_sample(datadir = './'):
    """Generates a list of globular cluster initial conditions randomly
    orbiting a dwarf galaxy.

    Positional Arguments:

    Keyword Arguments:

    """
    # Eric Andersson, 2018-01-15
    from . import read
    from . import constants
    from . import compute
    import numpy as np

    # Constants
    G = constants.gravitational_constant(unit = 'kpc(kpc/Myr)^2Msun')
    K = constants.conversion(unitchange = 'kpc/Myr->km/s')

    # Read set-up parameters.
    (N_GC, r_init, M_M31, M_GC, M_s, r_s, r_GC, r_fc, s) = read.setup(
            param = ['N_GC', 'r_init', 'M_M31', 'M_GC', 'M_s', 'r_s',
                'r_GC', 'r_fc', 'sigma_dv'],
            datadir = datadir)
    N_GC = int(N_GC)

    # Globular cluster orbital distances.
    rmin_GC = r_GC / compute.roche_lobe(1, M_GC, M_s)
    rmax_GC = compute.roche_lobe(r_fc, M_s, M_M31)

    # Generate positions of the globular clusters.
    sphere, N_GC = compute.sample_sphere((rmin_GC, rmax_GC), N_GC,
            dist = 'random')

    rp = np.zeros([N_GC, 3])
    for i in range(N_GC):
        rp[i] = (sphere[i][0], sphere[i][1], sphere[i][2])


    # Generate plane to determine direction of orbital velocity.
    nhat = np.zeros((N_GC, 3))
    mhat = np.zeros((N_GC, 3))
    khat = np.zeros((N_GC, 3))

    for i in range(N_GC):
        nhat[i] = np.array((rp[i][0], rp[i][1], rp[i][2]))
        nhat[i] /= np.linalg.norm(nhat[i])
        mhat[i] = compute.perpendicular_vector(nhat[i])
        mhat[i] /= np.linalg.norm(mhat[i])
        khat[i] = np.cross(nhat[i], mhat[i])

    # Set up orbital velocity.
    vp = np.zeros([N_GC, 3])
    for i in range(N_GC):
        dv = s*np.random.randn()
        theta = 2*np.pi * np.random.random()
        vrot = compute.plummer_vrot(np.linalg.norm(rp[i]), dv, M_s, r_s)
        vp[i] = (vrot * (np.sin(theta)*mhat[i] + np.cos(theta)*khat[i]))


    # Write to file
    data = open(datadir + 'GC_sample.txt', 'a+')
    for i in range(N_GC):
        data.write("%4.5f\t%4.5f\t%4.5f\t%3.5f\t%3.5f\t%3.5f\n"
                % (rp[i][0], rp[i][1], rp[i][2],
                   vp[i][0], vp[i][1], vp[i][2]))

def encounters(datadir = './'):
    """Generates a set of encounters for Galactic Potential Particle
    Integrator.

    Keyword Arguments:
        datadir
            Directory to store data.
    """
    # Eric Andersson, 2018-01-15
    from . import read
    from . import compute
    from . import constants
    from . import draw
    import numpy as np
    import os

    # Constants
    G = constants.gravitational_constant(unit = 'kpc(kpc/Myr)^2Msun')
    K = constants.conversion(unitchange = 'kpc/Myr->km/s')

    # Read in set-up parameters for encounters.
    (N, N_GC, r_init, theta_min, theta_max, phi_min, phi_max, traj,
        R_max, b_min, M_M31, M_s) = read.setup(param = ['N_set', 'N_GC',
            'r_init', 'theta_min', 'theta_max', 'phi_min', 'phi_max',
            'trajectory', 'R_max', 'b_min', 'M_M31', 'M_s'],
            datadir = datadir)
    N_GC = int(N_GC)
    N = int(N)

    # Generate dwarf galaxy initial conditions.
    count = 0
    while count < N:
        # Draw initial paramters.
        theta = draw.incoming_angle((theta_min*np.pi/180,
            theta_max*np.pi/180))
        phi = draw.impact_angle((phi_min*np.pi/180, phi_max*np.pi/180))
        v_inc = draw.incoming_velocity(traj, r_init)
        b = draw.impact_parameter(v_inc, R_max, b_min)

        # Retry if impact parameter is unreasonable large.
        if b > 1000:
            continue

        # Print encounter paramters to file.
        myFile = open('Encounters.txt', 'a')
        myFile.write('Encounter {0:03d}\n'.format(count))
        myFile.write('Incoming angle = {0:.1f} deg\n'.format(
            theta*180/np.pi))
        myFile.write('Impact angle = {0:.1f} deg\n'.format(
            phi*180/np.pi))
        myFile.write('Incoming velocity = {0:.1f} km/s\n'.format(v_inc))
        myFile.write('Impact paramter = {0:.1f} kpc\n'.format(b))
        myFile.close()

        # Position of impact parameter origin.
        r = r_init * np.array((np.cos(theta), 0, np.sin(theta)))

        # Set up plane for the impact paramter direction.
        g = r / np.linalg.norm(r)
        k = compute.perpendicular_vector(g)
        k /= np.linalg.norm(k)
        h = np.cross(g, k)
        h /= np.linalg.norm(k)

        # Impact parameter.
        b = b * (np.sin(phi)*h + np.cos(phi)*k)

        # Satellite position
        rs = b + r

        # Incoming velocity direction.
        vs = - v_inc * (r / np.linalg.norm(r))

        # Estimate orbital paramters
        (a, e, xa, xp) = compute.orbital_parameters(rs, vs, M_M31, M_s)
        myFile = open('Encounters.txt', 'a')
        myFile.write('Semi-major axis = {0:.2f}\n'.format(a))
        myFile.write('Eccentricity = {0:.4f}\n'.format(e))
        myFile.write('Pericentre distance = {0:.2f} kpc\n'.format(xp))
        myFile.write('Apocentre distance = {0:.2f} kpc\n\n'.format(xa))
        myFile.close()

        # Set up cluster population.
        rp = np.zeros([N_GC, 3])
        vp = np.zeros([N_GC, 3])

        # Read cluster sample data.
        sample = read.GC_sample(datadir = datadir)

        rp[...,0] = rs[0] + sample[..., 0]
        rp[...,1] = rs[1] + sample[..., 1]
        rp[...,2] = rs[2] + sample[..., 2]

        vp[...,0] = vs[0] + sample[..., 3]
        vp[...,1] = vs[1] + sample[..., 4]
        vp[...,2] = vs[2] + sample[..., 5]

        # Set up direcory of encounter and write to file.
        filename = './RUN{0:03d}/initial_conditions.txt'.format(count)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        # Overwrite existing data.
        myFile = open(filename, 'w')
        myFile.write("%4.5f\t%4.5f\t%4.5f\t%3.5f\t%3.5f\t%3.5f\n"
                    % (rs[0], rs[1], rs[2], vs[0], vs[1], vs[2]))
        myFile.close()

        # Append cluster data.
        myFile = open(filename, 'a')
        for i in range(N_GC):
            myFile.write("%4.5f\t%4.5f\t%4.5f\t%3.5f\t%3.5f\t%3.5f\n"
                    % (rp[i][0], rp[i][1], rp[i][2],
                       vp[i][0], vp[i][1], vp[i][2]))

        myFile.close()
        count += 1
