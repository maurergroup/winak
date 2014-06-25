# thctk.Convenience
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2004 Christoph Scheurer
#
#   This file is part of thctk.
#
#   thctk is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   thctk is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
###############
# CONVENIENCE #
###############
by Daniel Strobusch

This module provides tools to perform VSCF and VCI calculations easily with
a minimum of user input. The methods here will also exemplify a standard work
flow for a VSCF/VCI calculation.

The three major classes are:
    'Setup' to setup a coordinate system and a grid
    'VSCFSpectrum' to compute and print VSCF energies for all fundamental modes.
    'VCISpectrum' to perform a VCI calculation.
"""

from itertools import izip, count
import string
from operator import itemgetter

from thctk.numeric import *
from thctk.numeric.Rotation import rigidBodySuperposition
from thctk.Output import Section
from thctk.constants import eh2cm, mass

from thctk.QD.InternalCoordinates import ValenceCoordinateGenerator
from thctk.QD.InternalCoordinates import icSystem
from thctk.QD.NormalModes import NormalModes, massWeightedModes, momentOfInertia
from thctk.QD.NormalModes import centerOfMass
from thctk.QD.Coordinates import RectilinearCoordinates, InternalCoordinates
from thctk.QD.Coordinates import InternalEckartFrameCoordinates
from thctk.QD.Coordinates import InternalNormalModes
from thctk.QD.Coordinates import EckartConditions

def IC(atoms, x0, internals = None, masses = None):
    """
    Construct an 'icSystem' automatically.
    'masses' may be taken from the library.
    """
    try:
        ic = icSystem(internals, len(atoms), masses = masses, xyz = x0.copy())
    except TypeError:
        internals = ValenceCoordinateGenerator(atoms, masses = masses)(x0)
        ic = icSystem(internals, len(atoms), masses = masses, xyz = x0.copy())
    return ic

def coordinates(atoms, x0, hessian, masses = None, eckart = False,
                principalAxisSystem = True, ic = True, internals = None):
    """
    Construct everything coordinate related automatically.
    """
    x0 = x0.copy()
    if masses is None: # get masses from atoms
        masses = N.array([mass[string.capitalize(i)] for i in atoms])
    if len(masses) != len(atoms): # check shape of mass array
        masses = masses[::3]
    masses = N.asarray(masses)
    assert masses.shape == (len(atoms),)

    (ev, (omega, modes, redm)) = massWeightedModes(hessian, masses)
    omega = omega[6:]
    modes = modes[6:]
    if eckart:
        modes = EckartConditions(modes, x0, masses)

    if principalAxisSystem:
        x0 = (x0.reshape((-1, 3)) - centerOfMass(x0, masses)).ravel()
        I = momentOfInertia(x0, masses)
        (i, O) = N.linalg.eigh(I)
        l = zip(i, O.T)
        l.sort(key = itemgetter(0))
        a = l[0][1]
        a /= N.sign(a.take(N.where(N.max(N.abs(a)) == N.abs(a))[0])[0])
        b = l[2][1]
        b /= N.sign(b.take(N.where(N.max(N.abs(b)) == N.abs(b))[0])[0])
        c = N.cross(a, b)
        c/= N.linalg.norm(c)
        q = rigidBodySuperposition(N.array([a, b, c]), N.identity(3))[2][1]

        x0 = q.rotate(x0.reshape((-1, 3)).T).T.ravel()
        for i in range(len(modes)):
            modes[i] = q.rotate(modes[i].reshape((-1, 3)).T).T.ravel()
    ic = IC(atoms, x0, internals = internals, masses = masses) if ic else None

    if ic is not None:
        return InternalNormalModes(x0, masses, L = modes.T, ic = ic,
                                      atoms = atoms, freqs = omega)
    else:
        return RectilinearCoordinates(x0, masses, L = modes.T,
                                      atoms = atoms, freqs = omega)

    # return NormalModes(freqs = omega, modes = modes, unit = 1,
    #                    masses = masses, redm = None, x0 = x0, atoms = atoms,
    #                    ic = ic)


