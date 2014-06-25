# thctk.QD.NormalModes
# -*- coding: ascii -*-
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2002-2005 Christoph Scheurer
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
Module for handling Normal Modes
"""

import math, types
from UserList import UserList
from thctk.QC.Output import BohrToAngstrom
from thctk.QC import AtomInfo
import thctk.numeric.Roots
from thctk.numeric import *
from thctk.numeric.Rotation import rigidBodySuperposition
from thctk.constants import UNIT
from copy import copy, deepcopy
LA = importLinearAlgebra()

pi = math.pi

au2cm = 4.5563353e-6

def massWeightedModes(hessian, masses, tfact = 2.642461e+07):
    n, m = hessian.shape
    G = N.zeros(n, nxFloat)
    j = 0
    for m in masses:
        G[j:j+3] = 1.0/m
        j += 3
    G = N.sqrt(G)
    A = hessian * G[NewAxis,:]
    A *= G[:,NewAxis]
    ev = LA.Heigenvectors(A)
    omega = N.sqrt(abs(ev[0])*tfact)*N.sign(ev[0])
    modes = ev[1]*G
    redm = 1./N.sum(modes*modes, 1)
    return ev, (omega, modes, redm)

def centerOfMass(x, masses):
    x = x.reshape((-1, 3))
    if len(masses) == 3*len(x): masses = masses[::3]
    return N.sum(masses[:,N.newaxis]*x, axis = 0)/N.sum(masses)

def momentOfInertia(x, masses):
    x = x.reshape((-1, 3))
    if len(masses) == 3*len(x): masses = masses[::3]
    I = N.empty((3,3))
    I[0,0] = sum(masses[i]*(x[i][1]*x[i][1] + x[i][2]*x[i][2])
                    for i in range(len(x)))
    I[1,1] = sum(masses[i]*(x[i][0]*x[i][0] + x[i][2]*x[i][2])
                    for i in range(len(x)))
    I[2,2] = sum(masses[i]*(x[i][0]*x[i][0] + x[i][1]*x[i][1])
                    for i in range(len(x)))
    I[0,1] = I[1,0] = -sum(masses[i]*x[i][0]*x[i][1] for i in range(len(x)))
    I[0,2] = I[2,0] = -sum(masses[i]*x[i][0]*x[i][2] for i in range(len(x)))
    I[1,2] = I[2,1] = -sum(masses[i]*x[i][1]*x[i][2] for i in range(len(x)))
    return I


class NormalMode(UserArray):

    def __init__(self, mode, freq = 0, redm = None, fc = None, index = None,
                 masses = None, ic = None, x0 = None, signConvention = False,
                 munit = 1822.888485, funit = 1/219474.6):
        """
        The parameter unit = m/hbar accounts for the correct units of mass and
        frequency. The default value for unit is for the case where
        frequencies are given in cm^-1 and all other units are atomic units:

            1 Hartree = 219474.6 cm^-1
            1 Bohr = 0.52917721 Ang
            1 a.m.u. = 1822.888485 me
            hbar = 1

        convention: maximum displacement is positive
        """
        UserArray.__init__(self, mode)
        if signConvention:
            ma = max(mode)
            mi = min(mode)
            if mi < 0 and abs(mi) > abs(ma):
                self.array *= -1
        self.munit = munit
        self.funit = funit
        self.f = freq
        self.m = redm
        self.fc = fc
        self.i = index
        self.masses = masses

    def __repr__(self):
        return self.__class__.__name__ + "("+repr(self.f)+")"

    def __lt__(self,other):
        return self.f <  other.f

    def __le__(self,other):
        return self.f <= other.f

    def __eq__(self,other):
        return self.f == other.f

    def __ne__(self,other):
        return self.f != other.f

    def __gt__(self,other):
        return self.f >  other.f

    def __ge__(self,other):
        return self.f >= other.f

    def massWeightedProjection(self, other):
        return N.sum(self*other*N.sqrt(self.masses*other.masses))

    mdot = massWeightedProjection

    def dot(self, other):
        return N.dot(self, other)

    def norm(self):
        return N.sqrt(self.dot(self))

    def mnorm(self):
        return N.sqrt(self.mdot(self))

    def effectiveMassAndForceConstant(self, update = False, unit = 2*pi):
        if self.m is None or update:
            self.m = self.mdot(self)/self.dot(self)
        if self.fc is None or update:
            self.fc = self.m*(unit*self.f)**2   # fc = V''(0)/2
        return self.m, self.fc

    MC = effectiveMassAndForceConstant

    def GaussWidth(self, unit = None, mass = None):
        """
        The harmonic oscillator ground state wave function is proportional to
        exp(-1/2 m omega x^2 / hbar), where omega^2 = V''(0)/m . This is
        of the form exp(-1/2 a^2 x^2), where a^2 = m/hbar omega . The extrema
        of the first excited HO state are at \pm 1/a .
        """
        if unit is None: unit = self.munit * self.funit
        if mass is None: mass = self.m
        return 1/N.sqrt(unit*mass*self.f)        # this is 1/a

    def finiteDiffStep(self, V0, eps = 1.0e-6, m = None, unit = None, hbar = 1):
        """
        Optimal step size for finite difference approximation to the
        PES derivative around the minimum based on Normal Mode properties.

        See: Numerical Receipes in Fortran 2nd ed., Chap. 5.7
             (p. 181, eq. 5.7.5)

            h ~ sqrt(eps) . sqrt(V(0) / V''(0))
            h ~ sqrt(eps V(0)  m/2 ) . GaussWidth^2 / hbar

            hbar = 1
        """
#       print "freq in cm-1: %6.4f" %(219474.6/self.GaussWidth()**2) # freq in cm-1
#       k = (1./self.GaussWidth()**4)* self.m
#       print N.sqrt(k/self.m) * 219474.6  # f = sqrt(k/m) * au2cm gives freq in cm-1
#       hopt = N.sqrt((0.5 * eps * abs(V0))/ k)
        g = self.GaussWidth(mass = m, unit = unit)
        if m is None: m = self.m
        return N.sqrt((eps * abs(V0) * m)/2) * g*g/hbar

class NormalModes(UserList):

    def __init__(self, freqs, modes, unit = 2*pi, masses = None, redm = None,
                 fc = None, ic = None, x0 = None, atoms = None, V0 = 0,
                 signConvention = True, sort = False,
                 ):
        """
        unit    ... frequency unit (default unit = 2*pi)
        NOTE: For an unclear reason one might get into trouble, if two
              instances for 'NormalModes' point to the same ic instance.
              In this case use a deepcopy of ic to initialize the NormalModes
              object.
        """
        self.data = []
        self.unit = unit
        self.V0 = V0
        self.modes = N.asarray(modes, nxFloat)
        self.nmodes, self.ncrd = self.modes.shape
        try:
            self.freqs = N.asarray(freqs, nxFloat)
            if self.freqs.shape != (self.nmodes,):
                raise IndexError
        except:
            self.freqs = N.arange(self.nmodes).astype(nxFloat)
        self.natoms = self.ncrd / 3
        if atoms is None: atoms = ('X',) * self.natoms
        self.atoms = atoms
        ano = []
        for a in self.atoms:
            ano.append(AtomInfo.sym2no(a))
        self.ano = N.array(ano).astype(nxInt)
        if redm is None: redm = [None,]*self.nmodes
        else: self.redm = N.asarray(redm, nxFloat) # reduced masses
        if fc is None: fc = [None,]*self.nmodes
        for i in range(self.nmodes):
            self.append(NormalMode(self.modes[i], self.freqs[i], redm[i], fc[i], i,
                signConvention = signConvention))
        if masses is None: self.masses = N.ones(self.ncrd, nxFloat)
        else: self.setMasses(masses)
        self.setIC(ic)
        self.setx(x0)
        if sort: self.sort()

    def setMasses(self, m, amu = 1, f = None):
        """
        f       ... frequency scaling factor (default use self.unit)
        amu     ... mass unit (default m = 1)
        """
        if len(m) == self.ncrd:
            self.masses = amu * N.asarray(m, nxFloat)
        elif len(m) == self.natoms:
            mm = N.zeros(self.ncrd, nxFloat)
            k = 0
            for i in range(self.natoms):
                mm[k] = mm[k+1] = mm[k+2] = m[i]
                k += 3
            self.masses = amu * mm
        else: raise IndexError
        if f is None: f = self.unit
        for nm in self:
            nm.masses = self.masses
            nm.MC(update = 1, unit = f)

    def setx(self, x0 = None):
        if x0 is None:
            x0 = N.zeros(self.ncrd, nxFloat)
            self.q0 = None
        else:
            if self.ic is not None:
                self.ic.xyz[:] = x0
                self.q0 = self.ic().copy()
        self.x0 = N.ravel(x0)

    def setIC(self, ic = None):
        self.ic = ic
        if ic is not None:
            n = len(ic)
            if not hasattr(ic, 'xyz'): ic.xyz = N.zeros(self.ncrd, nxFloat)
            if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(n, nxFloat)
            self.ic.initA()
            for m in self:
                m.icMode = N.zeros(n, nxFloat)
                ic.B(m.array, m.icMode)
            if hasattr(self, 'q0'):
                if self.q0 is not None:
                    self.q0 = self.ic().copy()
            else:
                self.q0 = self.ic().copy()

    def icDisplacement(self, displacement = (), x0 = None, debug = False,
            **biArgs):
        self.biConvergence = None
        if x0 is None: self.ic.xyz[:] = self.x0
        else:          self.ic.xyz[:] = x0
        q = self.tmp_q
        q[:] = self.q0
        self.dq = dq = N.zeros(len(self)) # actual displacement
        if type(displacement) == types.ListType or \
           type(displacement) == types.TupleType:
            for i, eta in displacement:
                q += eta * self[i].icMode
                dq[i] = eta # store displacement
        elif type(displacement) == nxArrayType:
            for i in range(len(displacement)):
                q += displacement[i] * self[i].icMode
                dq[i] = displacement[i]
        else:
            raise TypeError("unknown displacement type")
        if not biArgs.has_key('initialize'):
            biArgs['initialize'] = 0
        try:
            self.biConvergence = self.ic.backIteration(q, **biArgs)
        except ValueError:
            if self.ic.backIteration == self.ic.sparseBackIteration:
                print("Sparse back iteration did not converge, " +
                      "retrying with dense back iteration")
                try:
                    if x0 is None: self.ic.xyz[:] = self.x0
                    else:          self.ic.xyz[:] = x0
                    self.biConvergence = self.ic.denseBackIteration(q, **biArgs)
                except ValueError:
                    print("Back iteration did not converge while displacing " +
                            "at: ", displacement)
                    raise
            else:
                print("Back iteration did not converge while displacing at: ",
                        displacement)
                raise
        return self.ic.xyz

    def cartDisplacement(self, displacement = (), x0 = None):
        if x0 is None: x0 = self.x0
        x = self.x0.copy()
        self.dq = dq = N.zeros(len(self)) # actual displacement
        if type(displacement) == types.ListType or \
           type(displacement) == types.TupleType:
            for i, eta in displacement:
                x += eta * self[i]
                dq[i] = eta
        elif type(displacement) == nxArrayType:
            for i in range(len(displacement)):
                x += displacement[i] * self[i]
                dq[i] = displacement[i]
        else:
            raise TypeError("unknown displacement type")
        try: self.ic.xyz[:] = x
        except AttributeError: pass

        return x

    def energyEvaluator(self, displacement, energy = None, cartesian = True,
            x0 = None, unit = N.sqrt(1822.888485)):
        d = Displace(self, displacement, cartesian = cartesian, x0 = x0,
            bohr = True, unit = 1/unit)
        return EnergyEvaluator(d, energy = energy, bohr = True)

    def visualizeXYZ(self, displace = (), scale = 1, unit = 1,
        steps = 11, file = "normal-modes.xyz", cartesian = True):

        cfmt = `self.natoms` + "\nstep: %3d   dq = %9.4f\n"
        xfmt = "%-4s%8.3f%8.3f%8.3f\n"
        if cartesian:
            disp = self.cartDisplacement
        else:
            disp = self.icDisplacement
        d = N.zeros(self.nmodes, nxFloat)
        for i in displace:
            d[i] = self[i].GaussWidth()
        grid = N.arange(steps) - 0.5*(steps - 1)
        grid *= 2.*scale/(steps - 1)/unit
        f = open(file, 'w')
        for i in range(steps):
            f.write( cfmt % (i, grid[i]) )
            x = N.reshape(disp(grid[i]*d), (self.natoms, 3))
            x *= BohrToAngstrom
            for j in range(self.natoms):
                f.write( xfmt % (self.atoms[j], x[j,0], x[j,1], x[j,2]) )
        f.close()

    def dumpMolden (self, filename="freq.molden"):
        """
        Dump a MOLDEN formatted file,
        mainly intended for visualizing a few selected modes
        """
        mlstr = "[Molden Format]\n[Atoms] Angs\n"
        tmp = "[FR-COORD] Bohrs\n"
        for i in xrange(self.natoms):
            charge = AtomInfo.Symbols.index(self.atoms[i])
            l = self.x0[i*3:(i*3)+3] * BohrToAngstrom
            mlstr += "%s  %5i %4i   %20.10f%20.10f%20.10f\n" \
                %(self.atoms[i], i+1, int(charge), l[0], l[1], l[2])
            l *= (1./BohrToAngstrom)
            tmp += "%s  %20.10f%20.10f%20.10f\n" \
                %(self.atoms[i], l[0], l[1], l[2])
        mlstr += tmp
        mlstr += " [FREQ]\n"
        tmp = "[FR-NORM-COORD]\n"
        for i in xrange(len(self.freqs)):
            mlstr += "%7.2f\n" %(self.freqs[i])
            tmp += " Vibration %20i\n" %(i+1)
            for j in xrange(self.natoms):
                l = self.modes[i][j*3:(j*3)+3]
                tmp += " %16.8f%16.8f%16.8f\n" %(l[0], l[1], l[2])
        mlstr += tmp
        file = open(filename, 'w')
        file.write(mlstr)
        file.close()

    def regularize(self, inveps = None, printInfo = True):
        """
        Create a new regularized basis. (by Max Blazejak, modified by DS)
        Works in place (copying is not possible at the moment)
        """
        from thctk.numeric.Matrix import regularizedInverse
        assert self.ic is not None
        # TODO: it is not possible to make a deepcopy, since the elements
        # of self are 'NormalMode' instances, which are derived from user
        # array. So at the moment it is not possible here, to create a new
        # basis as a copy.
        newBase = self

        L = N.array( [ i.icMode for i in newBase] ).T # modes as columns
        # build regularized inverse
        if inveps is not None:
            Ln = regularizedInverse(L, eps = inveps)
            Ln = regularizedInverse(Ln, eps = inveps)
        else: # use default
            Ln = regularizedInverse(L)
            Ln = regularizedInverse(Ln)
        for i in range(len(newBase)):
            newBase[i].icMode = Ln[:,i]
        if printInfo:
            print("Maximal deviation of regularized basis is %2.1e."
                                                    %(N.abs(Ln - L).max()))
        return newBase

class Displace:
    def __init__(self, modes, displacement, cartesian = False, x0 = None,
            bohr = True, unit = UNIT, debug = False, biArgs = {},
            alignToX0 = True):
        self.alignToX0 = alignToX0
        self.modes = modes
        self.masses = modes.masses[::3]
        self.ano = modes.ano
        self.unit = float(unit)
        self.dim = 0
        self.debug = debug
        if cartesian:
            self.displace = modes.cartDisplacement
            biArgs = {}
        else:
            self.displace = modes.icDisplacement
        self.biArgs = biArgs
        self.x0 = x0
        if type(displacement) == types.ListType or \
        type(displacement) == types.TupleType:
            self.displacement = []
            for d in displacement:
                try: # if d is a sequence
                    if len(d) != 2:
                        raise IndexError
                    else:
                        disp = (int(d[0]), float(d[1]))
                except TypeError: # if d is a scalar
                    disp = (int(d), 1.)
                self.displacement.append(disp)
        elif type(displacement) == types.IntType:
            self.displacement = ((displacement, 1),)
        elif type(self.displacement) == nxArrayType:
            d = []
            for i in range(len(displacement)):
                eta = displacement[i]
                if eta != 0:
                    d.append((i, eta))
            self.displacement = d
        else:
            raise TypeError("unknown displacement type")
        self.dim = len(self.displacement)
        n = self.modes[0].shape[0]

    def __len__(self):
        return self.dim

    def __call__(self, *v):
        v = N.asarray(v, dtype = N.float).ravel()
        assert len(v) == self.dim, (
            "Displacement vector '%s' has wrong lenght (expected %i)."
            %(str(v), int(self.dim)))
        d = []
        for k in range(self.dim):
            t = v[k]
            i, eta = self.displacement[k]
            d.append((i, eta*t*self.unit))
        xyz = self.displace(d, x0 = self.x0, **self.biArgs)
        if self.alignToX0:
            # xTmp = xyz.copy()
            modes = self.modes
            xRef = modes.ic.xRef if hasattr(modes.ic, 'xRef') else modes.x0
            assert xRef is not None
            (xyz, rmsd, trafo) = rigidBodySuperposition(xyz.reshape((-1, 3)),
                                               xRef.reshape((-1, 3)),
                                               weights = self.masses,
                                               # RMSD=True
                                               )
            xyz = xyz.ravel()

            # dq = self.modes.dq
            # from thctk.numeric.Rotation import *
            # q1 = Quaternion((dq[0]*1e-2, N.array([1., 0., 0.])))
            # q2 = Quaternion((dq[1]*1e-2, N.array([0., 1., 0.])))
            # q3 = Quaternion((dq[2]*1e-2, N.array([0., 0., 1.])))
            # q = q1*q2*q3
            # from pdb import set_trace
            # set_trace()
            # xyz = q.rotate(xyz.reshape((-1, 3)).T).T.ravel()

            # # PAS
            # masses = self.masses
            # xyz = (xyz.reshape((-1, 3)) - centerOfMass(xyz, masses)).ravel()
            # I = momentOfInertia(xyz, masses)
            # (e, O) = N.linalg.eigh(I)
            # l = zip(e, O.T)
            # from operator import itemgetter
            # l.sort(key = itemgetter(0))
            # a = l[0][1]
            # a *= N.sign(a.take(N.where(N.max(N.abs(a)) == N.abs(a))[0])[0])
            # b = l[2][1]
            # b *= N.sign(b.take(N.where(N.max(N.abs(b)) == N.abs(b))[0])[0])
            # c = N.cross(a, b)
            # c/= N.linalg.norm(c)
            # q = rigidBodySuperposition(N.array([a, b, c]), N.identity(3))[2][1]
            # xyz = q.rotate(xyz.reshape((-1, 3)).T).T.ravel()

            try: self.modes.ic.xyz[:] = xyz
            except AttributeError: pass
        return xyz

