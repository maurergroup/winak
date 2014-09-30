# thctk.Coordinates
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
# COORDINATES #
###############
by Daniel Strobusch

This module provides classes for different kinds of coordinates, i.e.
rectilinear and curvilinear coordinates.
The basis classes should also be used to derive classes for custom coordinate
systems. This will ensure, that the kinetic energy module can deal with the
coordinates properly.
"""

from warnings import warn

from INTERNALS.curvilinear.numeric import *
from INTERNALS.constants import UNIT
from INTERNALS.curvilinear.InternalCoordinates import icSystem, normalizeIC
from INTERNALS.curvilinear.numeric.Rotation import rigidBodySuperposition
from INTERNALS.curvilinear.numeric.Quaternions import Quaternion
from INTERNALS import AtomInfo
import INTERNALS.curvilinear._intcrd as intcrd

from operator import itemgetter

def centerOfMass(x, masses):
    """
    taken from QD.normalmodes
    """
    x = x.reshape((-1, 3))
    if len(masses) == 3*len(x): masses = masses[::3]
    return N.sum(masses[:,N.newaxis]*x, axis = 0)/N.sum(masses)

def df(f, x, order = 1, h = 1.0e-3):
    """
    Numerical derivative of arbitrary order using central differences for
    scalar and vector valued functions.
    """
    if order == 0: return N.asarray(f(x))
    hh = 2*h
    x = N.asarray(x)
    xph = x.copy()
    xmh = x.copy()
    shape = N.asarray(f(x)).shape
    fxp = N.empty(x.shape*(order - 1) + shape)
    fxm = N.empty(x.shape*(order - 1) + shape)
    g = N.empty(x.shape*order + shape)
    for i in range(len(x)):
        xph[i] += h
        xmh[i] -= h
        fxp[...] = df(f, xph, order - 1, h)
        fxm[...] = df(f, xmh, order - 1, h)
        g[i] = (fxp -fxm)/hh
        xph[i] = xmh[i] = x[i]
    return g

def applySignConvention(L, Li = None):
    """
    Apply sign convention on 'L', where 'L' contains modes as columns.
    (The convention is also applied on the inverse of 'L', 'Li'.)
    The sign convention is, that the max(abs(L[:,i]) = max(L[:,i]).
    """
    for i in xrange(L.shape[1]):
        if N.abs(N.min(L[:,i])) > N.max(L[:,i]):
            L[:,i] = -L[:,i]
            if Li is not None: Li[i,:] = -Li[i,:]

def EckartConditions(L, x0, masses):
    """
    Apply rotational Eckart conditions to modes (given as matrix L).
    """
    assert x0.ndim == 1
    if len(L) != len(x0): L = L.T
    if len(masses) != len(x0): masses = N.array([(masses,)*3]).T.ravel()
    basis = Coordinates(x0, masses)
    assert N.allclose(basis.x, x0)
    basis.evalAtrans()
    basis.evalArot()
    L = N.concatenate((basis.Atrans, basis.Arot, L), axis = 1)
    Li = (masses[:,N.newaxis]*L).T
    L = N.linalg.inv(Li)
    return L[:,6:].T
    # Li = (masses[:,N.newaxis]*L).T
    # q0 = x0 * masses
    # t = N.zeros((3, len(x0)))
    # r = N.zeros((3, len(x0)))
    # for i in range(0, len(x0), 3):
    #     t[:,i:i+3] = N.diag(masses[i:i+3])
    #     r[:,i:i+3] = N.array([N.cross(q0[i:i+3], j) for j in N.identity(3)])
    # return N.linalg.inv(N.concatenate((t, r, Li)))[:,6:].T


class Coordinates(object):
    """
    Basis class for coordinate classes. Derived classes must implement
    methods 's2x' and 'x2s'.
    """
    def __init__(self, x0, masses, ns = None, internal = False, atoms = None,
        freqs = None):
        """
        Construct coordinate object.
        x0 : array
            reference geometry for displacements.
        masses : array
            masses of atoms
        ns : int
            number of vibrational modes
        internal : bool, optional
            If 'True', coordinates are internal, which is important for the
            evaluation of the kinetic energy operator.
        atoms : sequence, optional
            A sequence of strings defining the atoms.
        freqs : sequence, optional
            A sequence giving approximate harmonic frequencies. Important for
            grid guess procedures.
        """
        self.atoms = atoms
        if atoms is not None:
            self.ano = N.array([AtomInfo.sym2no(i) for i in atoms],
                               dtype = N.int)
        else:
            self.ano = None

        self.freqs = freqs

        self.x0 = x0.copy()
        self.nx = nx = len(x0)
        if ns is None: ns = nx - 6
        if len(masses) == nx // 3:
            masses = N.array([(masses,)*3]).T.ravel()
        self.masses = masses
        self.ns = ns
        self.internal = internal
        self.x = N.empty(nx) # Cartesian coordinates
        self.s = N.empty(ns) # vibrational coordinates
        self.B = B = N.empty((6 + ns, nx)) # full B matrix
        self.Btrans = B[:3,:] # translational B matrix
        self.Brot = B[3:6,:] # rotational B matrix
        self.Bvib = B[6:,:] # vibrational B matrix
        self.A = A = N.empty((nx, 6 + ns)) # full A matrix
        self.Atrans = A[:,:3] # translational A matrix
        self.Arot = A[:,3:6] # rotational A matrix
        self.Avib = A[:,6:] # vibrational A matrix

        self.x[:] = x0 # init Cartesian coordinates

        Btrans = self.Btrans
        for i in range(0, nx, 3):
            Btrans[:,i:i+3] = N.diag(masses[i:i+3])
        Btrans /= N.sum(masses[::3])

        Atrans = self.Atrans
        for i in range(0, nx, 3):
            Atrans[i:i+3,:] = N.identity(3)

    def __len__(self):
        return self.ns

    def __getitem__(self, i):
        # Dummy code to be compatible with old NormalModes class.
        warn("'__getitem__' method does not return any meaningful data.",
                                                    DeprecationWarning)
        return N.zeros(1)

    def getDisplaceInstance(self, displacement, biArgs = None, x0 = None):
        """
        Get a 'Displace' instance for this coordinate object.
        """
        return Displace(self, displacement, biArgs = biArgs, x0 = x0)

    def getS(self, x = None):
        """
        Convert Cartesian coordinates to vibrational coordinates.
        """
        if x is not None: self.x[:] = x
        self.x2s()
        return self.s

    def getX(self, s = None):
        """
        Convert vibrational coordinates to Cartesian coordinates.
        """
        if s is not None: self.s[:] = s
        self.s2x()
        return self.x

    def evalB(self, out = None):
        """
        Evaluate and return full B matrix, i.e. the derivatives of
        translational, rotational and vibrational coordinates w.r.t. Cartesian
        coordinates.
        """
        if out is None:
            B = self.B
        else:
            B = out
            B[:3,:] = self.Btrans
        self.evalBrot(out = B[3:6,:])
        self.evalBvib(out = B[6:,:])
        return B

    def evalA(self, out = None):
        """
        Evaluate and return full A matrix, i.e. the derivatives of Cartesian
        coordinates w.r.t. translational, rotational and vibrational
        coordinates.
        """
        if out is None:
            A = self.A
        else:
            A = out
            A[:,:3] = self.Atrans
        self.evalArot(out = A[:,3:6])
        self.evalAvib(out = A[:,6:])
        return A

    def evalBtrans(self, out = None):
        """
        Evaluate and return translational B matrix, i.e. the derivatives of
        translational coordinates w.r.t. Cartesian coordinates.
        """
        if out is None:
            return self.Btrans
        else:
            out[:] = self.Btrans
            return out

    def evalAtrans(self, out = None):
        """
        Evaluate and return translational A matrix, i.e. the derivatives of
        Cartesian coordinates w.r.t. translational coordinates.
        """
        if out is None:
            return self.Atrans
        else:
            out[:] = self.Atrans
            return out

    def evalBrot(self, out = None):
        """
        Evaluate and return rotational B matrix, i.e. the derivatives of
        rotational coordinates w.r.t. Cartesian coordinates.
        """
        raise AttributeError('Not implemented!')

    def evalArot(self, out = None):
        """
        Evaluate and return rotational A matrix, i.e. the derivatives of
        Cartesian coordinates w.r.t. rotational coordinates.
        """
        if out is None: Arot = self.Arot
        else:           Arot = out
        x = self.x
        for i in range(0, self.nx, 3):
            xi = x[i:i+3]
            Arot[i:i+3,:] = N.array([
                                [     0,  xi[2], -xi[1]],
                                [-xi[2],      0,  xi[0]],
                                [ xi[1], -xi[0],      0]])
        return Arot

    def evalBvib(self, out = None, h = 1e-4):
        """
        Evaluate and return vibrational B matrix, i.e. the derivatives of
        vibrational coordinates w.r.t. Cartesian coordinates.
        (Default is numerical differentiation)
        """
        if out is None: Bvib = self.Bvib
        else:           Bvib = out
        x = self.x
        # compute numerical derivatives
        Bvib[:] = df(self.getS, x, order = 1, h = h).T*self.unit
        self.getS(x) # restore s, which was modified in df
        return Bvib

    def evalAvib(self, out = None, h = 1e-2):
        """
        Evaluate and return vibrational A matrix, i.e. the derivatives of
        Cartesian coordinates w.r.t. vibrational coordinates.
        (Default is numerical differentiation)
        """
        if out is None: Avib = self.Avib
        else:           Avib = out
        s = self.s
        x = self.x.copy() # save current x
        # compute numerical derivatives
        Avib[:] = df(self.getX, s, order = 1, h = h).T/self.unit
        self.getS(x) # restore x
        return Avib

    def x2s(self):
        """
        This method must be implemented by a derived class. The purpose of
        this method is to convert the geometry stored in 'self.x' to vibrational
        coordinates. The result must be stored in 'self.s'.
        The scheme for the implementation of this method is:
        self.s[:] = 'do something with self.x'
        """
        raise AttributeError('Not implemented!')

    def s2x(self):
        """
        This method must be implemented by a derived class. The purpose of
        this method is to convert the geometry stored in 'self.s' to Cartesian
        coordinates. The result must be stored in 'self.x'.
        The scheme for the implementation of this method is:
        self.x[:] = 'do something with self.s'
        """
        raise AttributeError('Not implemented!')

class EckartFrameCoordinates(Coordinates):
    """
    A coordinate class, to align all Cartesian coordinates constructed
    from vibrational coordinates to the Eckart frame.
    """
    def __init__(self, x0, masses, ns = None, internal = False,
                 atoms = None, freqs = None, xRef = None):
        """
        xRef : array, optional
            Reference geometry for the Eckart frame definition, by default 'x0'
            will be taken as reference. However, in rare cases, it is convenient
            to define a different geometry for the Eckart frame.
        """
        Coordinates.__init__(self, x0, masses, ns = ns, internal = internal,
                                atoms = atoms, freqs = freqs,)
        nx = self.nx
        self.xRef = xRef = xRef if xRef is not None else x0
        w = self.masses/N.sum(masses[::3])
        self.xw = xw = xRef*w
        xwX = N.sum(xw[0::3])
        xwY = N.sum(xw[1::3])
        xwZ = N.sum(xw[2::3])
        F = N.empty((nx//3, 3, 3))
        f = N.zeros((3,3))
        for i in xrange(0, nx, 3):
            # The sign of the quaternion parametrization is already included in
            # 'f'. Formally, there should be a factor of two
            # in front of 'f', but this factor cancels with another factor of
            # two in 'M' (see 'evalBrot').
            f[1,0] =  2*(xw[i+2] - w[i]*xwZ)
            f[2,0] = -2*(xw[i+1] - w[i]*xwY)
            f[2,1] =  2*(xw[i+0] - w[i]*xwX)
            f[0,1] = -f[1,0]
            f[0,2] = -f[2,0]
            f[1,2] = -f[2,1]
            F[i//3,:,:] = f
        self.F = F
        self.M = N.empty((3,3))

    __init__.__doc__ = Coordinates.__init__.__doc__.strip() +  __init__.__doc__

    def evalBrot(self, out = None):
        """
        Compute rotational B matrix analytically via Quaternions.
        """
        if out is None: Brot = self.Brot
        else:           Brot = out
        x = self.x
        xw = self.xw
        rr0 = 2*N.dot(x, xw)

        xx0 = -2*N.dot(x[0::3], xw[0::3])
        yy0 = -2*N.dot(x[1::3], xw[1::3])
        zz0 = -2*N.dot(x[2::3], xw[2::3])

        xy0 = -N.dot(x[0::3], xw[1::3])
        xz0 = -N.dot(x[0::3], xw[2::3])
        yz0 = -N.dot(x[1::3], xw[2::3])

        yx0 = -N.dot(xw[0::3], x[1::3])
        zx0 = -N.dot(xw[0::3], x[2::3])
        zy0 = -N.dot(xw[1::3], x[2::3])

        F = self.F
        M = self.M
        for i in xrange(0, self.nx, 3):
            M[0,0] = rr0 + xx0
            M[1,1] = rr0 + yy0
            M[2,2] = rr0 + zz0
            M[0,1] = M[1,0] = xy0 + yx0
            M[0,2] = M[2,0] = xz0 + zx0
            M[1,2] = M[2,1] = yz0 + zy0
            # Formally, there should be a factor of two in front of 'M',
            # but this factor cancels, after inversion, with another implicit
            # factor of two in 'F'.
            Brot[:,i:i+3] = N.dot(N.linalg.inv(M), F[i//3])
        return Brot

    def alignToFrame(self):
        """
        Align the current geometry stored in 'self.x' to the Eckart frame.
        """
        xRef = self.xRef
        x = rigidBodySuperposition(self.x.reshape((-1, 3)),
                                   xRef.reshape((-1, 3)),
                                   weights = self.masses[::3],
                                   )[0]
        self.x[:] = x.ravel()

class PASCoordinates(Coordinates):
    """
    A coordinate class, to align all Cartesian coordinates constructed
    from vibrational coordinates to the principal axis system (PAS).
    """
    def __init__(self, x0, masses, ns = None, internal = False,
                 atoms = None, freqs = None):
        """
        """
        Coordinates.__init__(self, x0, masses, ns = ns, internal = internal,
                                atoms = atoms, freqs = freqs,)

    __init__.__doc__ = Coordinates.__init__.__doc__.strip() +  __init__.__doc__

    def evalBrot(self, out = None):
        """
        Compute rotational B matrix analytically
        (Meyer and Guenthard, JCP 49, 1510 (1968).
        """
        if out is None: Brot = self.Brot
        else:           Brot = out
        # don't modify self.Arot
        Arot = self.evalArot(out = N.empty(self.Arot.shape))
        masses = self.masses
        I = N.dot(Arot.T, masses[:,N.newaxis]*Arot)
        I = N.diag(I)
        Brot[0,0::3] = 0.
        Brot[1,1::3] = 0.
        Brot[2,2::3] = 0.
        x = self.x
        for i in xrange(0, self.nx, 3):
            Brot[0,i+1] = masses[i]*x[i+2]/(I[2]-I[1])
            Brot[0,i+2] = masses[i]*x[i+1]/(I[2]-I[1])
            Brot[1,i+0] = masses[i]*x[i+2]/(I[0]-I[2])
            Brot[1,i+2] = masses[i]*x[i+0]/(I[0]-I[2])
            Brot[2,i+0] = masses[i]*x[i+1]/(I[1]-I[0])
            Brot[2,i+1] = masses[i]*x[i+0]/(I[1]-I[0])
        return Brot

    def alignToFrame(self):
        """
        Align the current geometry stored in 'self.x' to the PAS.
        """
        x = self.x
        masses = self.masses
        # subtract center of mass
        x = x.reshape((-1, 3))
        x -= N.sum(masses[::3,N.newaxis]*x, axis = 0) / N.sum(masses[::3])
        x = x.ravel()
        # don't modify self.Arot
        Arot = self.evalArot(out = N.empty(self.Arot.shape))
        I = N.dot(Arot.T, masses[:,N.newaxis]*Arot)
        (e, v) = N.linalg.eigh(I)
        ev = zip(e, v.T)
        ev.sort(key = itemgetter(0))
        a = ev[0][1]
        b = ev[2][1]
        a *= N.sign(a.take(N.where(N.max(N.abs(a)) == N.abs(a))[0])[0])
        b *= N.sign(b.take(N.where(N.max(N.abs(b)) == N.abs(b))[0])[0])
        c = N.cross(a, b)
        # c /= N.linalg.norm(c)
        assert N.abs(N.linalg.norm(c) - 1.) < 1e-12
        q = rigidBodySuperposition(N.array([a, b, c]), N.identity(3))[2][1]
        x = q.rotate(self.x.reshape((-1, 3)).T).T.ravel()
        self.x[:] = x

class RectilinearCoordinates(Coordinates):
    """
    Coordinate class for rectilinear coordinates, e.g. normal modes.
    """
    def __init__(self, x0, masses, ns = None, internal = False,
                 atoms = None, freqs = None, L = None, Li = None, unit = UNIT,
                 signConvention = True):
        """
        L : array, optional
            coordinates as column vectors
        Li : array, optional
            left inverse to L
        unit : float, optional
            multiplicative scalar for the definition of the vibrational
            coordinates. Usually the conversion factor of 'u' to 'a.u.' is used.
        signConvention : bool, optional
            flag to determine, if the sign convention should be applied on
            'L' and 'Li'. (See 'applySignConvention' for details.)
        """
        if ns is None: ns = L.shape[1]
        Coordinates.__init__(self, x0, masses, ns = ns, internal = internal,
                            atoms = atoms, freqs = freqs,)
        if signConvention: applySignConvention(L, Li)
        self.L = L
        if Li is None:
            self.x[:] = x0
            self.evalAtrans()
            self.evalArot()
            LL = N.concatenate((self.Atrans, self.Arot, L), axis = 1)
            Li = N.linalg.inv(LL)[6:,:]
            assert N.allclose(N.dot(Li, L), N.identity(self.ns), 1e-10, 1e-10),(
                    "Cannot invert definition of coordinates, give inverse " +
                    "of 'L' explicitely!")
        self.Li = Li
        assert N.allclose(N.dot(Li, L), N.identity(self.ns), 1e-10, 1e-10),(
            "'Li' is not inverse to 'L'!")
        self.Avib = L
        self.Bvib = Li
        self.unit = unit
    __init__.__doc__ = Coordinates.__init__.__doc__.strip() +  __init__.__doc__

    def x2s(self):
        self.s[:] = N.dot(self.Li, self.x - self.x0)/self.unit

    def s2x(self):
        self.x[:] = N.dot(self.L, self.s*self.unit) + self.x0

    def evalBvib(self, out = None):
        if out is None: return self.Bvib
        out[:] = self.Bvib
        return out

    def evalAvib(self, out = None):
        if out is None: return self.Avib
        out[:] = self.Avib
        return out

class CurvilinearCoordinates(Coordinates):
    """
    Abstract class for arbitrary curvilinear coordinates.
    """
    def __init__(self, x0, masses, ns = None, atoms = None, freqs = None):
        Coordinates.__init__(self, x0, masses, ns = None, atoms = atoms,
                                freqs = freqs,)

class InternalCoordinates(Coordinates):
    """
    Class for internal curvilinear coordinates.
    """
    def __init__(self, x0, masses, ns = None, ic = None, internal = True,
                 atoms = None, freqs = None,
                 L = None, Li = None, unit = UNIT, rcond = 1e-10, biArgs = {}):
        """
        L : array, optional
            coordinates as column vectors
        Li : array, optional
            left inverse to L
        unit : float, optional
            multiplicative scalar for the definition of the vibrational
            coordinates. Usually the conversion factor of 'u' to 'a.u.' is used.
        rcond : float, optional
            control the pseudo inversion of 'L', see 'numpy.linalg.pinv'
        biArgs : dict, optional
            keyword arguments to be passed to the back iteration
        """
        if ns is None: ns = L.shape[1]
        Coordinates.__init__(self, x0, masses, ns = ns, internal = internal,
                                atoms = atoms, freqs = freqs, )
        self.L = L
        if Li is None:
            Li = N.array(N.linalg.pinv(L, rcond = rcond))
            assert N.allclose(N.dot(Li, L), N.identity(self.ns)), (
                "Coordinate definition could not be inverted! " +
                "Try to regularize it.")
        self.Li = Li
        self.ic = ic
        self.unit = unit
        self.biArgs = biArgs
        ic.xyz[:] = x0
        q = ic()
        try:
            normalizeIC(q, *ic.tmp_icArrays)
        except AttributeError:
            ic.biInit()
            normalizeIC(q, *ic.tmp_icArrays)
        self.q0 = q.copy()
        self.getS(x0)
    __init__.__doc__ = Coordinates.__init__.__doc__.strip() +  __init__.__doc__

    def x2s(self):
        ic = self.ic
        ic.xyz[:] = self.x
        q = ic()
        normalizeIC(q, *ic.tmp_icArrays)
        dq = q - self.q0
        if ic.torsions is not None:
            dq = intcrd.dphi_mod_2pi(dq, N.asarray(ic.torsions, N.int32))
        self.s[:] = N.dot(self.Li, dq)/self.unit

    def s2x(self):
        ic = self.ic
        q = self.q0 + N.dot(self.L, self.s*self.unit)
        ic.xyz[:] = self.x
        ic.backIteration(q, **self.biArgs)
        R = centerOfMass(ic.xyz, self.masses)
        for i in range(0, self.nx, 3): ic.xyz[i:i+3] -= R
        self.x[:] = ic.xyz

    def evalBvib(self, out = None):
        if out is None: Bvib = self.Bvib
        else:           Bvib = out
        self.ic.xyz[:] = self.x
        Bvib[:] = N.dot(self.Li, self.ic.evalB().full())
        return Bvib

    def normalize(self):
        assert self.masses is not None, ("'masses' is 'None'!")
        assert self.masses.shape == self.x.shape, ("'masses' have wrong shape!")
        assert self.L is not None, ("'L' is 'None'!")
        L = self.L
        Li = self.Li
        x = self.x
        B = self.evalBvib()
        G = N.dot(B, 1./self.masses[:,N.newaxis]*B.T)
        d = N.sqrt(N.diag(G))
        for i in range(len(L.T)):
            di = d[i]
            L[:,i] *= di
            Li[i,:] /= di


class InternalEckartFrameCoordinates(EckartFrameCoordinates,
                                     InternalCoordinates):
    """
    Class for internal curvilinear coordinates connected to the Eckart frame.
    """
    def __init__(self, x0, masses, ns = None, internal = True,  xRef = None,
                 atoms = None, freqs = None,
                 ic = None, L = None, Li = None, unit = UNIT, rcond = 1e-10,
                 biArgs = {}
                 ):
        """
        """
        EckartFrameCoordinates.__init__(self, x0, masses, ns = ns, xRef = xRef)
        InternalCoordinates.__init__(self, x0, masses, ns = ns, ic = ic,
                                        internal = internal, atoms = atoms,
                                        freqs = freqs, L = L, Li = Li,
                                        unit = unit, rcond = rcond, biArgs =
                                        biArgs)
    __init__.__doc__ = (Coordinates.__init__.__doc__.strip() +
                        EckartFrameCoordinates.__init__.__doc__[
                        len(Coordinates.__init__.__doc__.strip()):] +
                        InternalCoordinates.__init__.__doc__[
                        len(Coordinates.__init__.__doc__.strip()):].strip() +
                        __init__.__doc__
                       )

    def s2x(self):
        InternalCoordinates.s2x(self)
        self.alignToFrame()

#Konsti
class DelocalizedCoordinates(InternalEckartFrameCoordinates):
    def __init__(self, x0, masses,u=None, ns = None, internal = True,  xRef = None,
                 atoms = None, freqs = None,
                 ic = None, L = None, Li = None, unit = UNIT, rcond = 1e-10,
                 biArgs = {}
                 ):
        self.u=u
        if L is None or Li is None:
            Li=u
            L = N.dot(N.linalg.inv(N.dot(Li,Li.transpose())),Li)
            L = L.transpose()
        InternalEckartFrameCoordinates.__init__(self,x0,masses,ns=ns,internal=internal,xRef=xRef,atoms=atoms,freqs=freqs,
                                                ic=ic,L=L,Li=Li,unit=unit,rcond=rcond,biArgs=biArgs)

    def get_vectors(self):
        """Returns the delocalized internal eigenvectors as cartesian
        displacements. Careful! get_vectors()[0] is the first vector.
        If you want to interpret it as a matrix in the same way numpy does,
        you will have to transpose it first. They are normed so that
        the largest single component is equal to 1"""
        ss = self.s
        w=[]
        for i in range(0,len(ss)):
            worked=False
            tries=1
            fac=0.01
            while not worked:
                try:
                    ss=N.zeros(len(self.s))
                    ss[i]=fac
                    dd=(self.getX(ss)-self.x0).reshape(-1,3)
                    dd/=N.max(N.abs(dd))
                    w.append(dd)
                    worked=True
                except:
                    tries+=1
                    fac*=2
                    if tries>40:
                        raise ValueError('NO convergence')
        w=N.asarray(w)
        return w

    def get_delocvectors(self):
        """Returns the delocalized internal eigenvectors."""
        return self.u.transpose()

    def write_jmol(self,filename,constr=False):
        """This works similar to write_jmol in ase.vibrations."""
        fd = open(filename, 'w')
        wtemp=self.get_vectors()
        for i in range(len(wtemp)):
            fd.write('%6d\n' % (len(self.x0)/3))
            fd.write('Mode #%d, f = %.1f%s cm^-1 \n' % (i, i, ' '))
            for j, pos in enumerate(self.x0.reshape(-1,3)):
                fd.write('%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f \n' %
                     (self.atoms[j], pos[0], pos[1], pos[2],
                      wtemp[i,j, 0],wtemp[i,j, 1], wtemp[i,j, 2]))
        fd.close()




#REINI
class CompleteAdsorbateInternalEckartFrameCoordinates(InternalEckartFrameCoordinates):
    """
    Class for curvilinear coordinates that ideally describe a surface adsorbed molecule
    The internal coordinates are the same number as cartesian coordinates
    The first three coordinates are the 3 center-of-mass positions
    The second three are extrinsic Tait Bryan (nautical) angles in a zyx convention.
    The others are internal coordinates
    Hereby InternalEckartFrameCoordinates are used.
    """

    def __init__(self, x0, masses, atoms, xRef = None, freqs = None, ic = None, Ltrans = None,
                Lrot = None, Lvib = None, Lvibi=None, cell = None, unit = UNIT, rcond = 1e-10, \
                com_list = None):

        from INTERNALS.constants import UNIT
        if unit is None:
            self.unit = UNIT
        else:
            self.unit = unit
        if cell is None:
            self.cell = N.identity(3)
        else:
            self.cell = cell
        self.nx = nx = len(x0) # number of Cartesian (OUTSIDE MBE/SG) coords
        self.ns = ns = len(x0)
        self.masses = masses
        self.atoms = atoms
        self.q0 = N.zeros(Lvibi.shape[1]+6)
        self.x0 = x0.copy()
        self.x = N.empty(self.nx)
        self.q = N.empty(Lvibi.shape[1]+6)
        self.s = N.empty(self.ns)

        if xRef is None:
            self.xRef = x0.copy()
        else:
            self.xRef = xRef

        #TRANSLATION  project out com
        if Ltrans is None:
           self.Ltrans = (N.identity(3))#*N.sqrt(1./masses.sum()))
        else:
            self.Ltrans = Ltrans
        self.Ltransi = N.linalg.inv(self.Ltrans)
        self.celli = N.linalg.inv(self.cell)
        if com_list is None:
            self.com_list = N.array(range(len(atoms)))
        else:
            self.com_list = N.array(com_list)

        com = N.zeros(3)
        for ind in self.com_list:
            com += x0.reshape((-1,3))[ind] * masses[ind]
        com = com / masses[self.com_list].sum()
        self.q0[:3] = N.dot(com,self.celli)
        x0 = (x0.reshape((-1,3)) - centerOfMass(x0, masses)).flatten()

        #Rotation
        if Lrot is None:
            self.Lrot = (N.identity(3))#*N.sqrt(1./masses.sum()))
        else:
            self.Lrot = Lrot
        self.Lroti = N.linalg.inv(self.Lrot)

        qrot = rigidBodySuperposition(x0.reshape((-1, 3)),
                                   self.xRef.reshape((-1, 3)),
                                   weights = self.masses[:],
                                   )[2][1]
        self.q0[3:6] = qrot.getTaitBryanZYXAngles()

        #Internal Coordinates
        if ic is None:
            raise ValueError('Internal coordinates have to be specified')

        self.Lvibi = Lvibi
        self.coords_int = InternalEckartFrameCoordinates(x0, \
                                masses, ns=None, internal=True,\
                                xRef = self.xRef,atoms = atoms, freqs = freqs, ic=ic,\
                                L=Lvib, Li=Lvibi, unit=self.unit, rcond=1e-10, biArgs={})

        #should be normalized
        #self.coords_int.normalize()
        self.q0[6:] = self.coords_int.q0

        self.n_atoms = len(atoms)
        self.B = B = N.empty((6 + ns, nx)) # full B matrix
        self.Btrans = B[:3,:] # translational B matrix
        self.Brot = B[3:6,:] # rotational B matrix
        self.Bvib = B[6:,:] # vibrational B matrix
        self.A = A = N.empty((nx, 6 + ns)) # full A matrix
        self.Atrans = A[:,:3] # translational A matrix
        self.Arot = A[:,3:6] # rotational A matrix
        self.Avib = A[:,6:] # vibrational A matrix
        #Btrans = self.Btrans
        #for i in range(0, self.nx, 3):
            #Btrans[:,i:i+3] = np.diag(masses[i:i+3])
        #Btrans /= np.sum(masses[::3])
        Atrans = self.Atrans
        for i in range(0, self.nx, 3):
            Atrans[i:i+3,:] = N.identity(3)

    def __len__(self):
        return self.ns # number of coords

    def getX(self, s): # conversion from 's' to 'x' (curvilinear to Cartesian)

        x = N.zeros(self.nx)
        s = N.array(s)
        #get internals
        self.coords_int.s[:] = s[6:]
        self.coords_int.s2x()
        x = self.coords_int.x
        #get ROTATIONS
        phi, theta, psi = self.q0[3:6] + N.dot(self.Lrot, s[3:6] * self.unit)
        quart_a = Quaternion((-phi, N.array([1.,0.,0.])))
        quart_b = Quaternion((-theta, N.array([0.,1.,0.])))
        quart_c = Quaternion((-psi, N.array([0.,0.,1.])))
        quart = quart_c * quart_b * quart_a
        x = quart.rotate(x.reshape((-1, 3)).T).T.ravel()
        #get translation
        com = N.zeros(3)
        for ind in self.com_list:
            com += x.reshape((-1,3))[ind] * self.masses[ind]
        com = com / self.masses[self.com_list].sum()
        x = (x.reshape((-1,3)) - com).flatten()
        ##TRANSLATION around Center of Mass
        com = N.dot(self.q0[:3] + N.dot(self.Ltrans, s[:3]*self.unit),self.cell)
        x = (x.reshape((-1,3)) + com).flatten()
        self.x = x
        return x

    def getS(self, x): # conversion from 's' to 'x' (curvilinear to Cartesian)
        s = N.zeros(self.nx)
        x = N.array(x)
        self.q = N.zeros(self.Lvibi.shape[1]+6)
        #TRANSLATION
        com = N.zeros(3)
        for ind in self.com_list:
            com += x.reshape((-1,3))[ind] * self.masses[ind]
        com = com / self.masses[self.com_list].sum()
        x = (x.reshape((-1,3)) - com).flatten()
        self.q[:3] = N.dot(com, self.celli)
        s[:3] = N.dot(self.Ltransi, self.q[:3]-self.q0[:3]) / self.unit
        #ROTATION
        com = centerOfMass(x, self.masses)
        x = (x.reshape((-1,3)) - com).flatten()
        qrot = rigidBodySuperposition(x.reshape((-1, 3)),
                                   self.xRef.reshape((-1, 3)),
                                   weights = self.masses[:],
                                   )[2][1]
        self.q[3:6] = qrot.getTaitBryanZYXAngles()
        s[3:6] = N.dot(self.Lroti, self.q[3:6]-self.q0[3:6]) / self.unit
        #INTERNALS
        self.coords_int.x = x
        self.coords_int.x2s()
        self.q[6:] = self.coords_int.q0 + N.dot(self.coords_int.L, self.coords_int.s*self.coords_int.unit)
        s[6:]= self.coords_int.s
        return s

    def Q2S(self, q):
        s = N.zeros(self.nx)
        s[:3] = N.dot(self.Ltransi, q[:3]-self.q0[:3]) / self.unit
        s[3:6] = N.dot(self.Lroti, q[3:6]-self.q0[3:6]) / self.unit
        s[6:]= N.dot(self.coords_int.Li, q[6:]-self.q0[6:]) / self.coords_int.unit
        return s

class ReducedDimSurfaceCoordinates(CompleteAdsorbateInternalEckartFrameCoordinates):
    """
    This Coordinate system implements CompleteAdsInternalEckartFrameCoordinates,
    but the internal coords are only a subset, giving a reduced dimensional
    description of the system. The other coordinates are either default, or
    read in as mbe potentials.
    """

    def __init__(self, x0, masses, atoms, xRef = None, freqs = None, ic=None, Ltrans = None,
                Lrot = None, Lvib = None, Lvibi = None, cell = None, unit = UNIT, rcond = 1e-10, \
                com_list = None, use_mbecoords = False, active_coords = None, data_dir=None):

        CompleteAdsorbateInternalEckartFrameCoordinates.__init__(self, x0=x0, \
                                                             masses=masses, atoms=atoms, xRef = xRef, \
                                                             freqs = freqs, ic=ic, \
                                                             Ltrans = Ltrans, Lrot = Lrot, \
                                                             Lvib = Lvib, Lvibi=Lvibi, cell = cell, unit = unit, rcond = 1e-10, \
                                                             com_list = com_list)

        #per default is the dimensionality the same is in Compl.Ads.Int.EckartCoords
        if active_coords is None:
            self.active_coord_dict = {}
            for i in range(self.nx):
                self.active_coord_dict[i] = i
        else:
            self.active_coord_dict = active_coords

        self.ns = ns = len(self.active_coord_dict)
        self.s = N.empty(self.ns)
        nx = self.nx
        self.nimplicit = nx
        ####REDUCED DIM STUFF
        if data_dir is None:
            self.data_dir = './'
        else:
            self.data_dir = str(data_dir)
        if use_mbecoords:
            self.use_mbecoords = True
            #load operator_basis
            try:
                self.mbe_coords = self.load_mbe_coordinates(self.data_dir)
            except NotImplementedError:
                self.use_mbecoords = False
                self.mbe_coords = None
        else:
            self.use_mbecoords = False
            #coords are initialized without MBE terms
            self.mbe_coords = None

            self.B = B = N.empty((6 + ns, nx)) # full B matrix
            self.Btrans = B[:3,:] # translational B matrix
            self.Brot = B[3:6,:] # rotational B matrix
            self.Bvib = B[6:,:] # vibrational B matrix
            self.A = A = N.empty((nx, 6 + ns)) # full A matrix
            self.Atrans = A[:,:3] # translational A matrix
            self.Arot = A[:,3:6] # rotational A matrix
            self.Avib = A[:,6:] # vibrational A matrix
            #Btrans = self.Btrans
            #for i in range(0, self.nx, 3):
                #Btrans[:,i:i+3] = np.diag(masses[i:i+3])
            #Btrans /= np.sum(masses[::3])

            Atrans = self.Atrans
            for i in range(0, self.nx, 3):
                Atrans[i:i+3,:] = N.identity(3)

    def load_mbe_coordinates(self, data_dir):
        """
        This function loads coordinate parametrizations from files q<num>.nc
        """
        raise NotImplementedError('load_mbe_coordinates has to be implemented manually.')

    def modify_implicit_coords(self, ss, s):
        """
        If use_mbecoords is false, this function smoothes the output geometries and yields
        better structure guesses
        """
        raise NotImplementedError('modify_implicit_coords has to be implemented manually')

    def getX(self, s):

        ss = N.zeros(self.nx)
        for i in range(len(ss)):
            if i in self.active_coord_dict.keys():
                ss[i] = s[self.active_coord_dict[i]]
            else:
                if self.use_mbecoords:
                    ss[i] = self.mbe_coords[i](s)
                pass

        if not self.use_mbecoords:
            try:
                ss = self.modify_implicit_coords(ss, s)
            except NotImplementedError:
                pass

        x = CompleteAdsorbateInternalEckartFrameCoordinates.getX(self, ss)
        return x

    def getS(self, x):

        ss = CompleteAdsorbateInternalEckartFrameCoordinates.getS(self, x)

        s = N.zeros(len(self))
        for i in range(len(ss)):
            if i in self.active_coord_dict.keys():
                s[self.active_coord_dict[i]] = ss[i]
        return s


#Reini
class CompleteDelocalizedCoordinates(CompleteAdsorbateInternalEckartFrameCoordinates):
    """
    Derived CAIFC object do host Delocalized Internal Coordinates
    """
    def __init__(self, x0, masses, u=None, xRef = None, atoms=None, freqs = None, ic = None, Ltrans = None,
                Lrot = None, cell = None, unit = UNIT, rcond = 1e-10, \
                com_list = None):
        if u is None:
            raise ValueError('the DI vector u has to be given!')
        self.u = u
        Li = u
        L = N.dot(N.linalg.inv(N.dot(Li,Li.transpose())),Li)
        L = L.transpose()
        CompleteAdsorbateInternalEckartFrameCoordinates.__init__(self, x0=x0, \
                                                                masses=masses, atoms=atoms, xRef = xRef, \
                                                                freqs = freqs, ic=ic, \
                                                                Ltrans = Ltrans, Lrot = Lrot, \
                                                                Lvib = L, Lvibi=Li, cell = cell, unit = unit, rcond = 1e-10, \
                                                                com_list = com_list)
    def get_vectors(self):
        """Returns the delocalized internal eigenvectors as cartesian
        displacements. Careful! get_vectors()[0] is the first vector.
        If you want to interpret it as a matrix in the same way numpy does,
        you will have to transpose it first. They are normed so that
        the largest single component is equal to 1"""
        ss = self.s
        w=[]
        for i in range(0,len(ss)):
            worked=False
            tries=1
            fac=0.01
            while not worked:
                try:
                    ss=N.zeros(len(self.s))
                    ss[i]=fac
                    dd=(self.getX(ss)-self.x0).reshape(-1,3)
                    dd/=N.max(N.abs(dd))
                    w.append(dd)
                    worked=True
                except:
                    tries+=1
                    fac*=2
                    if tries>40:
                        raise ValueError('NO convergence')
        w=N.asarray(w)
        return w

    def get_DI_vectors(self):
        """TBD"""
        ss = self.s
        w=[]
        for i in range(6,len(ss)):
            ss=N.zeros(len(self.s))
            ss[i]=1
            dd=(self.getX(ss)-self.x0).reshape(-1,3)
            dd/=N.max(N.abs(dd))
            w.append(dd)
        w=N.asarray(w)
        return w

    def get_trans_vectors(self):
        """TBD"""
        ss = self.s
        w=[]
        for i in range(0,3):
            ss=N.zeros(len(self.s))
            ss[i]=1
            dd=(self.getX(ss)-self.x0).reshape(-1,3)
            dd/=N.max(N.abs(dd))
            w.append(dd)
        w=N.asarray(w)
        return w

    def get_rot_vectors(self):
        """TBD"""
        ss = self.s
        w=[]
        for i in range(3,6):
            ss=N.zeros(len(self.s))
            ss[i]=1
            dd=(self.getX(ss)-self.x0).reshape(-1,3)
            dd/=N.max(N.abs(dd))
            w.append(dd)
        w=N.asarray(w)
        return w

    def get_delocvectors(self):
        """Returns the delocalized internal eigenvectors."""
        return self.u.transpose()

    def write_jmol(self,filename,constr=False):
        """This works similar to write_jmol in ase.vibrations."""
        fd = open(filename, 'w')
        wtemp=self.get_vectors()
        for i in range(len(wtemp)):
            fd.write('%6d\n' % (len(self.x0)/3))
            fd.write('Mode #%d, f = %.1f%s cm^-1 \n' % (i, i, ' '))
            for j, pos in enumerate(self.x0.reshape(-1,3)):
                fd.write('%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f \n' %
                     (self.atoms[j], pos[0], pos[1], pos[2],
                      wtemp[i,j, 0],wtemp[i,j, 1], wtemp[i,j, 2]))
        fd.close()





class InternalPASCoordinates(PASCoordinates, InternalCoordinates):
    """
    Class for internal curvilinear coordinates connected to the PAS.
    """
    def __init__(self, x0, masses, ns = None, internal = True,
                 atoms = None, freqs = None,
                 ic = None, L = None, Li = None, unit = UNIT, rcond = 1e-10,
                 biArgs = {}
                 ):
        """
        """
        PASCoordinates.__init__(self, x0, masses, ns = ns)
        InternalCoordinates.__init__(self, x0, masses, ns = ns, ic = ic,
                                        internal = internal, atoms = atoms,
                                        freqs = freqs, L = L, Li = Li,
                                        unit = unit, rcond = rcond, biArgs =
                                        biArgs)
    __init__.__doc__ = (Coordinates.__init__.__doc__.strip() +
                        PASCoordinates.__init__.__doc__[
                        len(Coordinates.__init__.__doc__.strip()):] +
                        InternalCoordinates.__init__.__doc__[
                        len(Coordinates.__init__.__doc__.strip()):].strip() +
                        __init__.__doc__
                       )

    def s2x(self):
        InternalCoordinates.s2x(self)
        self.alignToFrame()




class InternalNormalModes(InternalCoordinates):
    """
    Class for internal curvilinear normal modes.
    """
    def __init__(self, x0, masses, ns = None, ic = None, internal = True,
                 atoms = None, freqs = None,
                 L = None, Li = None, unit = UNIT, rcond = 1e-10, biArgs = {},
                 signConvention = True):
        """
        L : array, optional
            coordinates as column vectors
        Li : array, optional
            left inverse to L
        unit : float, optional
            multiplicative scalar for the definition of the vibrational
            coordinates. Usually the conversion factor of 'u' to 'a.u.' is used.
        rcond : float, optional
            control the pseudo inversion of 'L', see 'numpy.linalg.pinv'
        biArgs : dict, optional
            keyword arguments to be passed to the back iteration
        signConvention : bool, optional
            flag to determine, if the sign convention should be applied on
            'L' and 'Li'. (See 'applySignConvention' for details.)
        """
        assert ic is not None
        assert L is not None
        if signConvention: applySignConvention(L, Li)
        ic.xyz[:] = x0
        if ns is None: ns = L.shape[1]
        U = N.empty((len(ic), ns), order = 'F')
        for i in xrange(ns): ic.B(L[:,i], U[:,i])

        InternalCoordinates.__init__(self, x0, masses, ns = ns, ic = ic,
                                     internal = internal, atoms = atoms,
                                     freqs = freqs, rcond = rcond,
                                     biArgs = biArgs, unit = unit,
                                     L = U, Li = None)
    __init__.__doc__ = Coordinates.__init__.__doc__.strip() +  __init__.__doc__


class Displace:
    """
    Convenience class to get displaced Cartesian geometries from grid points.
    This class is compatible to the old Displace class from
    'thctk.QD.NormalModes'.
    """
    def __init__(self, coords, displacement, biArgs = None, x0 = None):
        assert isinstance(coords, Coordinates)
        self.x0 = coords.x0.copy() if x0 is None else x0
        self.coords = coords
        self.displacement = displacement
        self.biArgs = biArgs
        self.s = N.zeros(len(coords))
        self.ano = coords.ano
        self.masses = coords.masses

    def __len__(self):
        return len(self.displacement)

    def __call__(self, *v):
        v = N.asarray(v, dtype = N.float).ravel()
        assert len(v) == len(self), (
            "Displacement vector '%s' has wrong length (expected %i)."
            %(str(v), int(len(self))))

        s = self.s.copy()
        for (i, vi) in zip(self.displacement, v):
            s[i] += vi
        backsub = False
        if self.biArgs not in(None, {}):
            try:
                biArgs = self.coords.biArgs
                self.coords.biArgs = self.biArgs
                backsub = True
            except AttributeError:
                pass
        self.coords.x[:] = self.x0
        x = self.coords.getX(s)
        if backsub:
            self.coords.biArgs = biArgs
        return x

