#! /usr/bin/env python3
#=======================================================================
# plot.py
#
# Facilities for plotting the data from Galactic potential N-body
# integrator.
#
# Eric Andersson, 2018-01-11
#=======================================================================

def trajectories2D(plane, particles = '',
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
    if type(particles) == list:
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
    if type(particles) == list:
        for par in particles:
            plt.plot(x[par], y[par], '-',
                    label = 'particle_{}'.format(par))

    # Add legend
    if legend:
        plt.legend(loc = 'best', numpoints = 1)

    # Save figure
    if not plotdir == '':
        plt.savefig(plotdir + plane + '_trajectory.pdf')

    plt.show()

def separation(particles, run,
        satellite = True, datadir = './data/', plotdir = '',
        dontshow = False, filename = 'separation'):
    """Plots the separation between the particles and M31 as function of
    time for a given encounter.

    Positional Arguments:
        particles
            List of all particle numbers.
        run
            Encounter number

    Keyword Arguments:
        satellite
            Adds trajectory of satellite.
        datadir
            Directory of data.
        plotdir
            Function saves a pdf of figure in this directory if given.
        dontshow
            If True, figure will be cleared when finished.
        filename
            Namn of file if plot is saved.
    """
    # Eric Andersson, 2018-01-12
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from . import read
    from . import analyse


    # Read in data
    (_, x, y, z, vx, vy, vz, Ek, Ep) = read.particle(particles, datadir)
    (t, xs, ys, zs, _, _, _, _, _, _) = read.satellite(datadir)
    npar, nt = x.shape

    # Compute the distances to M31.
    # Dwarf galaxy
    rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # Globular clusters
    rp = np.zeros([npar, nt])
    vp = np.zeros([npar, nt])
    for par in particles:
        rp[par] = np.sqrt(x[par]**2 + y[par]**2 + z[par]**2)
        vp[par] = np.sqrt(vx[par]**2 + vy[par]**2 + vz[par]**2)

    # Count MGC1-like
    (MGC1, nMGC1, unb, nunb, ret, nret) = analyse.MGC1_like(particles, run, t = t,
            rs = rs, r = rp, v = vp, Ek = Ek, Ep = Ep,
            datadir = datadir)


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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Left plot
    ax1.minorticks_on()
    ax1.set_yscale('log')
    ax1.set_ylim(1e1, 1e4)
    ax1.set_xlim(-2, 12)
    ax1.set_xlabel(
            r'${\rm Time}\quad ({\rm Gyr})$', fontsize = 16)
    ax1.set_ylabel(
            r'${\rm Distance\ from\ M31}\quad ({\rm kpc})$',
            fontsize = 16)

    # Right plot
    ax2.minorticks_on()
    ax2.set_ylim(1e1, 1e4)
    ax2.set_xlim(6e-1, 1e3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(NullFormatter())
    ax2.set_xlabel(r'${\rm Number\ of\ Clusters}$', fontsize = 16)

    # Plot data

    # Globular clusters
    ncap = len(particles) - (nunb + nret)
    print('Total: {}'.format(len(particles)))
    print('Captured: {}'.format(ncap))
    print('Retained: {}'.format(nret))
    print('Unbound: {}'.format(nunb))
    print('MGC1-like: {}'.format(nMGC1))

    for par in particles:
        if MGC1[par]:
            ax1.plot(t/1000, rp[par], 'g-', lw = 0.5, alpha = 1)
        else:
            if unb[par]:
                ax1.plot(t/1000, rp[par], '-y', lw = 0.5, alpha = 1)
            elif ret[par]:
                ax1.plot(t/1000, rp[par], '-b', lw = 0.5, alpha = 1)
            else:
                ax1.plot(t/1000, rp[par], '-', color = 'grey',
                        lw = 0.5, alpha = 1)

    # Add labels
    ax1.plot([],[], '-', color = 'grey', lw = 1, alpha = 1,
            label = r'${\rm Captured}$')
    ax1.plot([],[], '-b', lw = 1, alpha = 1,
            label = r'${\rm Retained}$')
    ax1.plot([],[], '-y', lw = 1, alpha = 1,
            label = r'${\rm Unbound}$')
    ax1.plot([],[], 'g-', lw = 1, alpha = 1,
            label = r'${\rm MGC1-like}$')

    # Dwarf galaxy
    ax1.plot(t/1000, rs, 'r--', lw = 3, zorder = 10,
            label = r'${\rm Dwarf\ galaxy }$')

    # MGC1 orbital distance.
    ax1.plot(t/1000, 200*np.ones(t.size), 'k--', lw = 3, zorder = 10)

    # Histogram
    bins = np.logspace(np.log10(1e0), np.log10(1e4), 80)
    ax2.hist(rpf, bins=bins, orientation = 'horizontal',
            edgecolor = 'w', color = 'grey')
    ax2.hist(rpf[MGC1], bins=bins, orientation = 'horizontal',
            edgecolor = 'w', color = 'g')
    ax2.hist(rpf[unb], bins=bins, orientation = 'horizontal',
            edgecolor = 'w', color = 'y')
    ax2.hist(rpf[ret], bins=bins, orientation = 'horizontal',
            edgecolor = 'w', color = 'b')

    # Finilize figure
    xticks = ax1.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    ax1.legend(loc='upper left', frameon = False)
    if not plotdir == '':
        plt.savefig(plotdir + filename + '.pdf')
    if not dontshow:
        plt.show()
    if dontshow:
        plt.clf()

def IC_histogram(particles, param, bins = 30, datadir = './', plotdir = ''):
    """Plots a histogram of the distribution of the wanted paramter.

    Positional Arguments:
        particles
            List of particles that will be used.
        param
            Paramters that will be plotted.

    Keyword Arguments:
        bins
            Number of bins.
        datadir
            Directory of encounter information.
        plotdir
            Directory for saving plot.
    """
    # Eric Andersson, 2018-01-17
    from . import read
    import numpy as np
    import matplotlib.pyplot as plt

    # Read in the data for the wanted paramters.
    x = read.encounter(param, datadir = datadir)[particles]

    # Set up figure.
    fig = plt.figure()
    plt.minorticks_on()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'${}$'.format(param), fontsize = 16)
    plt.ylabel(r'${\rm Number\ of\ encounters}$', fontsize = 16)

    # Plot data
    plt.hist(x, bins=bins,
            edgecolor = 'w', color = 'royalblue')

    # Finilize figure
    if not plotdir == '':
        plt.savefig(plotdir + param + '_histogram.pdf')
    plt.show()

def dwarf_distribution2D(redist = False, proj_radius = 400,
        datadir = './', plotdir = ''):
    """Creates a plot of how incoming dwarf galaxies are projected on a
    spherical surface surounding the centre of M31.

    Positional Arguments:

    Keyword Arguments:
        redist
            If True, then points will be distributed in phi and on both
            hemispheres.
        proj_radius
            Radius of projected sphere.
        datadir
            Directory of simulation.
        plotdir
            If provide, figure is saved as pdf in plotdir directory.
    """
    # Eric Andersson, 2018-01-22.
    from . import read
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    # Initialize variable and set parameters.
    nenc = int(read.info('nruns', datadir = datadir))
    xs = np.zeros(nenc)
    ys = np.zeros(nenc)
    zs = np.zeros(nenc)

    # Read data
    for i in range(nenc):
        (t,x,y,z,_,_,_,_) = read.satellite(
                datadir = datadir + 'RUN{0:03d}/data/'.format(i))
        # Sanity check.
        if np.sqrt(x[0]**2 + y[0]**2 + z[0]**2) < proj_radius:
            raise ValueError("Dwarf initiated inside 400 kpc.")
        for j in range(x.size):
            if np.sqrt(x[j]**2 + y[j]**2 + z[j]**2) < proj_radius:
                xs[i], ys[i], zs[i] = x[j], y[j], z[j]
                break

    # Set up figure
    fig = plt.figure()
    ax = plt.subplot(111, projection="hammer")
    plt.grid(True)
    ax.xaxis.set_major_formatter(NullFormatter())

    # Compute longitude and latitude.
    r = np.sqrt(xs**2 + ys**2 + zs**2)
    theta = np.arcsin(zs/r)
    phi = np.arctan2(ys,xs)

    # Redistribute particles along phi \in (0,2pi) and change
    # distribution of theta from (0,pi/2) to (0,pi).
    if redist:
        phi += 2*np.pi*np.random.random(phi.size)
        mask = np.random.choice(a = [True, False], size = theta.size)
        theta[mask] = -theta[mask]

    # Fix periodic bounardy conditions.
    # Theta
    for i in range(theta.size):
        if theta[i] < -np.pi/2:
            theta[i] += np.pi
        elif theta[i] > np.pi/2:
            theta[i] -= np.pi
    # Phi
    for i in range(phi.size):
        if phi[i] > np.pi:
            phi[i] -= 2*np.pi

    # Plot data
    ax.plot(phi, theta, '.r', label = r'$\rm Dwarf$')

    # Save figure
    if not plotdir == '':
        plt.savefig(plotdir + 'dwarf_2Ddist.pdf')

    plt.show()

def dwarf_distribution3D(redist = False, proj_radius = 400,
        datadir = './', plotdir = ''):
    """Creates a plot of how incoming dwarf galaxies are projected on a
    spherical surface surounding the centre of M31.

    Positional Arguments:

    Keyword Arguments:
        redist
            If True, then points will be distributed in phi and on both
            hemispheres.
        proj_radius
            Radius of projected sphere.
        datadir
            Directory of simulation.
        plotdir
            If provide, figure is saved as pdf in plotdir directory.
    """
    # Eric Andersson, 2018-01-23.
    from . import read
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axis3d

    # Initialize variable and set parameters.
    nenc = int(read.info('nruns', datadir = datadir))
    xs = np.zeros(nenc)
    ys = np.zeros(nenc)
    zs = np.zeros(nenc)

    # Read data
    for i in range(nenc):
        (t,x,y,z,_,_,_,_) = read.satellite(
                datadir = datadir + 'RUN{0:03d}/data/'.format(i))
        # Sanity check.
        if np.sqrt(x[0]**2 + y[0]**2 + z[0]**2) < proj_radius:
            raise ValueError("Dwarf initiated inside 400 kpc.")
        for j in range(x.size):
            if np.sqrt(x[j]**2 + y[j]**2 + z[j]**2) < proj_radius:
                xs[i], ys[i], zs[i] = x[j], y[j], z[j]
                break

    # Set up figure.
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d',
        'aspect':'equal'})
    ax.grid(False)
    ax.set_xlim(-(proj_radius + 100), proj_radius + 100)
    ax.set_ylim(-(proj_radius + 100), proj_radius + 100)
    ax.set_zlim(-(proj_radius + 100), proj_radius + 100)
    ax.set_xlabel(r'$x,\,{\rm kpc}$', fontsize = 16)
    ax.set_ylabel(r'$y,\,{\rm kpc}$', fontsize = 16)
    ax.set_zlabel(r'$z,\,{\rm kpc}$', fontsize = 16)
    ax.scatter(0, 0, 0, '.k', marker = r'$\bigotimes$', c = 'k',
            s = 45, label = r'$\rm M31$', zorder = 10)

    # Add wireframe at proj_radius.
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    ax.plot_wireframe(
            proj_radius * np.outer(np.sin(theta), np.cos(phi)),
            proj_radius * np.outer(np.sin(theta), np.sin(phi)),
            proj_radius * np.outer(np.cos(theta), np.ones_like(phi)),
            color='grey', rstride=1, cstride=1, alpha = 0.4)

    # Redistribute particles along phi \in (0,2pi) and change
    # distribution of theta from (0,pi/2) to (0,pi).
    if redist:
        r = np.sqrt(xs**2 + ys**2 + zs**2)
        theta = np.arccos(zs/r)
        phi = np.arctan(ys/xs)

        # Redistribute in phi
        phi += 2*np.pi*np.random.random(phi.size)
        xs = r * np.sin(theta) * np.cos(phi)
        ys = r * np.sin(theta) * np.sin(phi)
        zs = r * np.cos(theta)
        # Redistribute in theta
        mask = np.random.choice(a = [True, False], size = theta.size)
        zs[mask] = -zs[mask]

    # Plot data
    ax.scatter(xs, ys, zs, s=10, c='r', zorder=9,
            label = r'$\rm Dwarf$')

    # Finialize figure
    ax.legend(loc = 'best', numpoints = 1, fancybox = True,
            framealpha = 0.3, scatterpoints = 1)

    # Save figure
    if not plotdir == '':
        plt.savefig(plotdir + 'dwarf_3Ddist.pdf')

    plt.show()
