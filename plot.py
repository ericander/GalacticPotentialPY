#! /usr/bin/env python3
#=======================================================================
# plot.py
#
# Facilities for plotting the data from Galactic potential N-body
# integrator.
#
# Eric Andersson, 2018-01-11
#=======================================================================

def trajectories2D(plane, particles,
        xlim = (-250, 250), ylim = (-250, 250), legend = True,
        satellite = True, datadir = './data/', plotdir = ''):
    """Plots the projected trajectory of the particles in the given
    plane.

    Positional Arguments:
        plane
            2D plane in which trajectories are projected.
        particles
            List of all particle numbers.

    Keyword Arguments:
        satellite
            Adds trajectory of satellite.
        datadir
            Directory of data.
        plotdir
            Function saves a pdf of figure in this directory if given.
    """
    # Eric Andersson, 2018-01-11
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from . import read

    # Read in data
    if plane == 'xy':
        (_, x, y, _, _, _, _) = read.particle(particles, datadir)
    elif plane == 'xz':
        (_, x, _, y, _, _, _) = read.particle(particles, datadir)
    elif plane == 'yz':
        (_, _, x, y, _, _, _) = read.particle(particles, datadir)
    else:
        raise ValueError(
            "Keyword plane only accepts 'xy', 'xz', or 'yz'. ")
    if satellite:
        if plane == 'xy':
            (_, xs, ys, _, _, _, _, col) = read.satellite(datadir)
        elif plane == 'xz':
            (_, xs, _, ys, _, _, _, col) = read.satellite(datadir)
        elif plane == 'yz':
            (_, _, xs, ys, _, _, _, col) = read.satellite(datadir)
        else:
            raise ValueError(
                "Keyword plane only accepts 'xy', 'xz', or 'yz'. ")

    # Set up figure.
    fig = plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if plane == 'xy':
        plt.xlabel(r'$x \quad {\rm [kpc]}$', fontsize = 16)
        plt.ylabel(r'$y \quad {\rm [kpc]}$', fontsize = 16)
    elif plane == 'xz':
        plt.xlabel(r'$x \quad {\rm [kpc]}$', fontsize = 16)
        plt.ylabel(r'$z \quad {\rm [kpc]}$', fontsize = 16)
    elif plane == 'yz':
        plt.xlabel(r'$y \quad {\rm [kpc]}$', fontsize = 16)
        plt.ylabel(r'$z \quad {\rm [kpc]}$', fontsize = 16)

    # Add luminous extent of M31.
    r_b_lum = read.info("r_b_lum")
    h_d_lum = read.info("h_d_lum")
    r_d_lum = read.info("r_d_lum")


    plt.plot(0,0, 'sk', markersize = 7, label = 'M31', zorder = 0)
    if plane == 'xz' or plane == 'yz':
        bulge = plt.Circle((0,0), r_b_lum, color='k', fill = True)
        plt.gcf().gca().add_artist(bulge)
        disc = patches.Rectangle((-0.5*r_d_lum, -0.5*h_d_lum),
                r_d_lum, h_d_lum, color = 'k')
        plt.gcf().gca().add_patch(disc)

    elif plane == 'xy':
        disc = plt.Circle((0,0), r_d_lum, color='k', fill = True)
        plt.gcf().gca().add_artist(disc)

    # Add satellite trajectory
    if satellite:
        col = (col == 1)
        noncol = (col == 0)
        plt.plot(xs[noncol], ys[noncol], '-', color = 'grey',
                label = 'Dwarf galaxy', lw = 2)
        plt.plot(xs[col], ys[col], '--', color = 'grey', lw = 2)

    # Plot particle trajectories.
    for par in particles:
        plt.plot(x[par], y[par], '-', label = 'particle_{}'.format(par))

    # Add legend
    if legend:
        plt.legend(loc = 'best', numpoints = 1)

    # Save figure
    if not plotdir == '':
        plt.savefig(plotdir + plane + '_trajectory.pdf')

    plt.show()

def separation(particles,
        satellite = True, datadir = './data/', plotdir = ''):
    """Plots the separation between the particles and M31 as function of
    time for a given encounter.

    Positional Arguments:
        particles
            List of all particle numbers.

    Keyword Arguments:
        satellite
            Adds trajectory of satellite.
        datadir
            Directory of data.
        plotdir
            Function saves a pdf of figure in this directory if given.
    """
    # Eric Andersson, 2018-01-12
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from . import read
    from . import analyse


    # Read in data
    (_, x, y, z, _, _, _) = read.particle(particles, datadir)
    (t, xs, ys, zs, _, _, _, _) = read.satellite(datadir)
    npar, nt = x.shape

    # Compute the distances to M31.
    # Dwarf galaxy
    rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # Globular clusters
    rp = np.zeros([npar, nt])
    for par in particles:
        rp[par] = np.sqrt(x[par]**2 + y[par]**2 + z[par]**2)

    # Count MGC1-like
    MGC1, _, _ = analyse.MGC1_like()

    # Compute final distance.
    rpf = np.zeros(npar)
    for par in particles:
        rpf[par] = rp[par][-1]

    # Set up figure
    fig = plt.figure(1, figsize=(12, 8))
    left_plot_size = [0.1, 0.1, 0.65, 0.85]
    right_plot_size = [0.75, 0.1, 0.2, 0.85]
    ax1 = plt.axes(left_plot_size)
    ax2 = plt.axes(right_plot_size)

    # Left plot
    ax1.minorticks_on()
    ax1.set_yscale('log')
    ax1.set_ylim(1e0, 1e4)
    ax1.set_xlabel(
            r'${\rm Time,}\ t\ [{\rm Gyr}]$', fontsize = 16)
    ax1.set_ylabel(
            r'${\rm Distance\ from\ M31,}\ r\ {\rm [kpc]}$',
            fontsize = 16)
    xticks = ax1.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)

    # Right plot
    ax2.minorticks_on()
    ax2.set_ylim(1e0, 1e4)
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(NullFormatter())
    ax2.set_xlabel(r'${\rm Number\ of\ Clusters}$', fontsize = 16)
    ax2.set_xscale('log')

    # Plot data
    # Dwarf galaxy
    ax1.plot(t/1000, rs, 'k--', lw = 3, zorder = 10,
            label = r'${\rm Dwarf\ galaxy }$')

    # Globular clusters
    ax1.plot(t[MGC1]/1000, rp[MGC1], 'g-', lw = 0.5, alpha = 1,
            label = r'${\rm MGC1-like}$')
    ax1.plot(t[(MGC1 == False)]/1000, rp[(MGC1 == False)], '-',
            color = 'grey', lw = 0.5, alpha = 1,
            label = r'${\rm Cluster}$')

    # MGC1 orbital distance.
    ax1.plot(t/1000, 200*np.ones(t.size), 'r--', lw = 3, zorder = 10,
            label = r'${\rm MGC1}$')

    # Histogram
    bins = np.logspace(np.log10(1e0), np.log10(1e4), 50)
    ax2.hist(rpf, bins=bins, orientation = 'horizontal',
            edgecolor = 'w', color = 'royalblue')

    # Finilize figure
    ax1.legend(loc='upper left')
    if not plotdir == '':
        plt.savefig(plotdir + 'separation.pdf')
    plt.show()

