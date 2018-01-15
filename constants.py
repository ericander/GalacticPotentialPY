#! /usr/bin/env python3
#=======================================================================
# constants.py
#
# Facilities for physical constants.
#
# Eric Andersson, 2018-01-14
#=======================================================================

def gravitational_constant(unit = 'SI'):
    """The Newtonian gravitational constant.

    Keyword Arguments:
        unit
            Unit in which G is returned.
    """
    import numpy as np

    if unit == 'SI':
        return 6.67408e-11
    elif unit == 'kpc(km/s)^2Msun':
        return 4.300912722e-6
    elif unit == 'kpc(kpc/Myr)^2Msun':
        return 4.558e-13 * np.pi**2
    else:
        raise NotImplementedError(unit + "is not implemented")

def powerlaw_density(obj):
    """Returns the parameters for a power-law density for an object.
    """
    if obj == 'M31':
        return 2759688103.0073333, 1.2297303185878623
    elif obj == '1e9Dwarf':
        return 41305269.86629897, 2.9999999999999996
    else:
        raise NotImplementedError(obj + "is not implemented yet")

def M31_mass():
    """Returns mass of M31 in solar masses.
    """
    return 1.4e12

def conversion(unitchange):
    """Returns conversion factor for unit change.

    Posisional Argument:
        unitchange
            From which unit to which unit.
    """
    if unitchange == 'kpc/Myr->km/s':
        return 978.5
    else:
        raise NotImplementedError(unitchange + " is not implemented.")

