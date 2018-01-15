#! /usr/bin/env python3
#=======================================================================
# compute.py
#
# Facilities for computing values for the Galactic potential particle
# integrator.
#
# Eric Andersson, 2018-01-14
#=======================================================================

def orbital_parameters(r, v, m1, m2):
    """Computes the orbital parameters for a 2-body orbit.

    Positional Arguments:
        r
            Position of satellite in reference system of m1 body.
        v
            Velocity of satellite in reference system of m1 body.
        m1
            Mass of major body.
        m2
            Mass of minor body.
    """
    # Eric Andersson, 2018-01-14
    import numpy as np
    from . import constants

    # Contants
    G = constants.gravitational_constant(unit = 'kpc(km/s)^2Msun')

    # Semi-major axis
    a = 1 / ((2/np.linalg.norm(r)) - (np.linalg.norm(v)**2 / (G * (m1 + m2))))

    # Eccentricity
    h = np.linalg.norm(np.cross(r, v))
    e = np.sqrt(1 - (h**2 / (G * (m1 + m2)*a )))

    # Apocentre
    xa = (1 + e)*a

    # Pericentre
    xp = (1 - e)*a

    return a, e, xa, xp

def tidal_radius(r, v, M_M31, Ms,
        xp = '', alpha = 1):
    """Computes the tidal radius for a satellite orbiting M31. Assumes
    both objects to have point mass potentials.

    Positional Arguments:
        r
            Separation between satellite and centre of potential.
        v
            Velocity of satellite.
        M_M31
            Mass of M31.
        Ms
            Mass of satellite galaxy.

    Keyword Arguments:
        xp
            Pericentre of satellite orbit.
        a
            Orbit orientation in satellite.
    """
    # Eric Andersson, 2018-01-14
    import numpy as np
    from . import read

    # Calculate orbital parameters.
    if xp == '':
        (a, e, xa, xp) = orbital_parameters(r, v, M_M31, Ms)
    else:
        (a, e, xa, _) = orbital_parameters(r, v, M_M31, Ms)

    # Compute tidal radius
    x = np.linalg.norm(r)
    L = a*(1 - e**2)
    rt = (x**4 * L * Ms / M_M31)**(1/3) * \
            ( (np.sqrt(alpha**2 + 1 + (2*x/L)) - alpha ) / \
            (L + 2*x) )**(2/3)
    return rt

def roche_lobe(A, m1, m2):
    """Computes the Roche lobe raduis for two masses m1 and m2.

    Positional Arguments:
        A
            Separation
        m1
            Mass of main object
        m2
            Mass of secular object
    """
    # Eric Andersson, 2018-01-14
    import numpy as np

    q = m1/m2
    r = A * (0.49*q**(2/3) / (0.6*q**(2/3) + np.log(1 + q**(1/3))))

    return r

def pointmass_potential(r, M):
    """ Computes the potential of a pointmass at distance r.

    Positional Arguments:
        r
            Distance
        M
            Mass of point
    """
    # Eric Andersson, 2018-01-14
    import numpy as np
    from . import constants

    # Constants
    G = constants.gravitational_constant(unit = 'kpc(km/s)^2Msun')

    return M*G / r
