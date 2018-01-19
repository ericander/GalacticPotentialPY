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


def perpendicular_vector(v):
    """ Function that given a vector generates another vector which
    is perpendicular to the given vector.

    Positional Arguments:
        v
            Vector
    """
    # Eric Andersson, 2018-01-15
    import numpy as np

    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

def sample_sphere(rlim, N, dist = 'random'):
    """ Function that generates N points inside a spherical shell with
    radius limited by rlim.

    Positional Arguments:
        rlim
            Minimin and maximum radius of the sphere.
        N
            Number of points on the sphere.
    Keyword Arguments:
        dist
            Distribution of the points.
    """
    # Eric Andersson, 2018-01-15
    import numpy as np

    if dist == 'equidistant':
        count=0
        a = 4*np.pi / N
        d = np.sqrt(a)
        M_theta = int(np.pi / d)
        d_theta = np.pi / M_theta
        d_phi = a / d_theta
        vec = np.zeros([N, 3])
        for m in range(M_theta):
            theta = np.pi * ( m + 0.5 ) / M_theta
            M_phi = int(2*np.pi*np.sin(theta) / d_phi)
            for n in range(M_phi):
                phi = 2*np.pi*n / M_phi
                r = np.random.uniform(rlim[0]**2, rlim[1]**2, 1)
                vec[count] = r * np.array([np.sin(theta)*np.cos(phi),
                                       np.sin(theta)*np.sin(phi),
                                       np.cos(theta)])
                count += 1
        print('Created ' + str(count) + ' points on the sphere.')
        return vec, int(count)

    elif dist == 'random':
        vec = np.random.randn(N, 3)
        for i in range(N):
            vec[i] /= np.linalg.norm(vec[i])
            vec[i] *= np.sqrt(
                    np.random.uniform(rlim[0]**2, rlim[1]**2, 1))
        return vec, int(N)
    else:
        raise ValueError(dist + " does not exist.")

def plummer_vrot(r, dv, M, r_c):
    """ Calculates the orbital velocity of a particle in a Plummer
    potential.

        Arguments:
            r
                Orbital radius.
            dv
                Small velocity change to heat the orbit slighly.
    """
    # Eric Andersson, 2018-01-15
    from . import constants
    import numpy as np

    # Constants
    G = constants.gravitational_constant(unit = 'kpc(km/s)^2Msun')

    # Calculate the gravitational acceleration.
    a = G * M * r / (r_c**2 + r**2)**(3/2)

    return np.sqrt(r * a) + dv

def M31_vrot(r):
    """The rotational velocity of the M31 galaxy in unit km/s.

    Positional Arguments:
        r
            Radial position in disc of M31.
    """
    # Eric Andersson, 2018-01-15
    from . import read
    from . import constants
    import numpy as np

    # Constants
    G = constants.gravitational_constant(unit = 'kpc(km/s)^2Msun')
    (M_b, r_b, M_d, a, b, V_h, r_c) = read.setup(
            param = ['M_b', 'r_b', 'M_d', 'A', 'B', 'V_h', 'r_h'])

    # Gravitational acceleration
    ab = G * M_b / (r_b + r)**2
    ad = G * M_d * r / ( (a + b)**2 + r**2 )**(3/2)
    ah = V_h**2 * r / (r_c**2 + r**2)

    return np.sqrt(r * (a_b + a_d + a_h))

def M31_potential(R, z):
    """Computes the potential of the M31 galaxy given cylindrical
    coordinates.

    Positional Arguments:
        R
            Radial position along the plane of the galactic disc.
        z
            Vertical postition perpendicular to the galactic disc.
    """
    # Eric Andersson, 2018-01-18
    from . import read
    from . import constants
    import numpy as np

    # Constants.
    G = constants.gravitational_constant(unit = 'kpc(km/s)^2Msun')

    # Read in M31 parameters.
    (M_b, r_b, M_d, A, B, V_h, r_h) = read.setup(['M_b', 'r_b', 'M_d',
            'A', 'B', 'V_h', 'r_h'], datadir = './')

    # Bulge potential.
    r = np.sqrt(R**2 + z**2)
    Phi_b = - G * M_b / (r_b + r)

    # Disc potential.
    Phi_d = - G * M_d / np.sqrt(R**2 + (A + np.sqrt(z**2 + B**2)**2))

    # Halo potential.
    Phi_h = 0.5 * V_h**2 * np.log(r_h**2 + r**2)

    return Phi_b + Phi_d + Phi_h
