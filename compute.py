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
    a = 1 / ((2/np.linalg.norm(r)) - (np.linalg.norm(v)**2 / \
            (G * (m1 + m2))))

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
    if r.shape[1] == 1:
        if xp == '':
            (a, e, xa, xp) = orbital_parameters(r, v, M_M31, Ms)
        else:
            (a, e, xa, _) = orbital_parameters(r, v, M_M31, Ms)
    else:
        if xp == '':
            (a, e, xa, xp) = orbital_parameters(r[...,0], v[...,0],
                    M_M31, Ms)
        else:
            (a, e, xa, _) = orbital_parameters(r[...,0], v[...,0],
                    M_M31, Ms)

    # Compute tidal radius
    if r.shape[1] == 1:
        x = np.linalg.norm(r)
    else:
        x = np.linalg.norm(r, axis = 0)
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
    a_b = G * M_b / (r_b + r)**2
    a_d = G * M_d * r / ( (a + b)**2 + r**2 )**(3/2)
    a_h = V_h**2 * r / (r_c**2 + r**2)

    return np.sqrt(r * (a_b + a_d + a_h))

def M31_potential(R, z, model = 'Bekki'):
    """Computes the potential of the M31 galaxy given cylindrical
    coordinates.

    Positional Arguments:
        R
            Position in along R direction.
        z
            Position along z axis.

    Keyword Arguments:
        model
            Which potenital model to use.
    """
    # Eric Andersson, 2018-01-18
    from . import read
    from . import constants
    import numpy as np

    # Constants.
    G = 4.558e-13 * np.pi**2

    # Translate coordinates.
    r = np.sqrt(R**2 + z**2)

    if model == 'Bekki':
        # Parameters.
        M_b = 9.2e10
        r_b = 0.7
        M_d = 1.3e11
        A = 6.5
        B = 0.26
        r_h = 12
        V_h = 186
        M_h = 1.24e12
        r_cut = 155.08
        Phi_0 = 208993.00328

        # Bulge potential.
        Phi_b = - G * M_b / (r_b + r)

        # Disc potential.
        Phi_d = - G * M_d / \
                np.sqrt(R**2 + (A + np.sqrt(z**2 + B**2))**2)

        # Halo potential.
        Phi_h = 0.5 * V_h**2 * np.log(r_h**2 + r**2) - Phi_0

        if r.size > 1:
            cut = (r > r_cut)
            Phi_h[cut] = - G * M_h / r[cut]
        elif r > r_cut:
            Phi_h = - G * M_h / r

        return Phi_b + Phi_d + Phi_h

    elif model == 'Geehan':
        # Parameters.
        M_b = 3.3e10
        r_b = 0.61
        M_d = 1.034365e11
        A = 6.42833
        B = 0.26476
        r_h = 8.18
        delta = 27e4
        rho = 136

        # Bulge potential.
        Phi_b = - G * M_b / (r_b + r)

        # Disc potential.
        Phi_d = - G * M_d / \
                np.sqrt(R**2 + (A + np.sqrt(z**2 + B**2))**2)

        # Halo potential.
        Phi_h = (-4 * np.pi * G * delta * rho * r_h**3 / r) * \
                np.log((r + r_h)/r_h)

        return Phi_b + Phi_d + Phi_h

    else:
        raise ValueError(model + ' is not available. Valid ' + \
            'models are Bekki or Geehan (not implemented).')

def M31_enclosed_mass(R, z, model = 'Bekki'):
    """ Computes the enclosed mass of the M31 galaxy given position.
    The function assumes the mass enclosed within a sphere with radius
    at the given position.

    Positional Arguments:
        R
            Distance along the plane of the galactic disc.
        z
            Vertical distance, perpendicular to R.

    Keyword Arguments:
        model
            The potential model that is used.
    """
    # Eric Andersson, 2018-02-19.
    import numpy as np
    from scipy import integrate
    from . import constants

    # Constants
    G = constants.gravitational_constant()

    # Set limits.
    r = np.sqrt(R**2 + z**2)
    Rlim = [0, R]
    zlim = [0, z]
    rlim = [0, r]

    # Set up model parameters.
    if model == 'Bekki':
        # Parameters.
        M_b = 9.2e10
        r_b = 0.7
        Md = 1.3e11
        a = 6.5
        b = 0.26
        r_h = 12
        V_h = 186
        M_h = 1.24e12
        r_cut = 155.08
        Phi_0 = 208993.00328

        # Bulge
        M_bulge = M_b * (r**2 / (r + r_b)**2)

        # Disc
        def f(R,z):
            return ((a*R**2 + (a + 3*np.sqrt(z**2 + b**2)) * \
            (a + np.sqrt(z**2 + b**2))**2) / \
            ( (R**2 + (a + np.sqrt(z**2 + b**2))**2)**(5/2) * \
            (z**2 + b**2)**(3/2) ))*R

        I = integrate.nquad(f, [Rlim, zlim])

        M_disc = (b**2 * Md / 2) * I[0]

        # Halo
        if r < 300:
            def f(r):
                return ((3*r_h**2 + r**2)*r**2) / (r_h**2 + r**2)**2

            I = integrate.nquad(f, [rlim])

            M_halo = V_h**2 / G * I[0]
        else:
            M_halo = M_h

    if model == 'Geehan':
        # Parameters.
        M_b = 3.3e10
        r_b = 0.61
        M_d = 1.0343e11
        A = 6.43
        B = 0.2647
        r_h = 8.8
        delta = 27e4
        rho = 140
        r_d = 5.4
        sigma = 4.6e8

        # Bulge
        M_bulge = M_b * r**2 / (r_b + r)**2

        # Disc
        M_disc = 2*np.pi*sigma*r_d**2*(1 - (1 + r/r_d)*np.exp(-r/r_d))

        # Halo
        M_halo = 4*np.pi*G*delta*rho*r_h**3 * (np.log((r + r_h)/r_h) \
                - r / (r + r_h))

    return M_bulge + M_disc + M_halo
