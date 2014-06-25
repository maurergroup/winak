# thctk.numeric.EigenSolver
# -*- coding: latin1 -*-
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2002-2006 Christoph Scheurer
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

import math
from thctk.numeric import *
from copy import copy 
LA = importLinearAlgebra()
from operator import itemgetter

try:
    from scipy.linalg.decomp import eigh
#   from symeig import symeig as eigh
except:
    pass

pi = math.pi

class Eigensolver:

    def __init__(self, H, nstates = None):
        self.H = H
        n = len(H)
        if nstates is None: nstates = n
        self.n = n
        self.nstates = nstates
        self.list = []

    def __len__(self):
        return len(self.list)

    def __call__(self, sym = False, overwrite = False):
        if sym:
            try:
                en, ev = eigh(self.H, overwrite_a = overwrite)
                ev = N.transpose(ev) # symeig & eigh vectors are in Fortran order
            except NameError:
                en, ev = LA.Heigenvectors(self.H)
            self.en = N.array(en)
            self.ev = N.array(ev)
        else:
            self.en, self.ev = LA.eigenvectors(self.H)
        self.makeList()
        return self

    def __getitem__(self, i):
        return self.list[int(i)]

    def E(self, i):
        return self.list[int(i)][0]

    def Psi(self, i):
        return self.list[int(i)][1]

    def idx(self, i):
        return self.list[int(i)][2]

    val = E
    vec = Psi

    def makeList(self, nstates = None):
        """
        sort and only keep the highest nstates eigenvectors (used to remove
        translational and rotational solutions)
        """
        if nstates is not None: self.nstates = nstates
        l = []
        for i in range(self.n): l.append((self.en[i], self.ev[i], i))
        l.sort(key = itemgetter(0))
        self.list = l[(self.n-self.nstates):]

class GaussianCollocation1D(Eigensolver):
    """
    Solving the vibrational Schrödinger equation with potential V using the
    collocation method of Yang and Peet, CPL 153 (1988), 98 with distributed Gaussians at
    points given by x. The parameter c determines the width of the Gaussian
    basis functions.
    """
    
    def __init__(self, x, w = None, c = 0.7, nstates = None, mass = 1,
                 keepMatrices = True):
        """
        'mass' may be a scalar (constant mass) or a vector (representation of 
        a non-constant reduced mass on the grid).
        """
        n = len(x)
        if nstates is None: nstates = n

        self.grid = x
        if w is not None:
            self.weights = w
        else:       # assume Simpson's rule
            self.weights = N.zeros(len(x), nxFloat)
            for i in range(1,len(x)-1):
                self.weights[i] = x[i+1] - x[i-1]
            self.weights[0]  = x[1]  - x[0]
            self.weights[-1] = x[-1] - x[-2]
            self.weights *= 0.5
        self.n = n
        self.nstates = nstates

        # generate parameters A (see eqs. (13) and (14) of the reference and
        # JCP 84 (1986), 306)
        A = N.zeros(n, nxFloat)

        cc = c*c
        A[0]  = cc / (x[1] - x[0])**2
        A[-1] = cc / (x[-1] - x[-2])**2
        cc *= 4
        for i in range(1,n-1):
            dx = x[i+1] - x[i-1]
            A[i] = cc / (dx*dx)

        # generate n distributed Gaussian wavefunctions R, 1st and 2nd
        # derivatives D and G
        R = N.zeros((n, n), nxFloat)
        D = N.zeros((n, n), nxFloat)
        G = N.zeros((n, n), nxFloat)
        S = N.zeros((n, n), nxFloat)

        Rd = (2/pi) * A
        Rd = N.power(Rd, 0.25, Rd)      # diagonal elements of R
        # self.Rd = Rd # DS: removed, no need for that

        for i in range(n):
            a = A[i]
            fR = Rd[i]
            fD = -2*a
            S[i,i] = 1
            R[i,i] = fR
            D[i,i] = 0
            G[i,i] = fD*fR
            for j in range(i):
                aj = A[j]
                dx = x[j] - x[i]
                dx2 = dx * dx
                aij = a*aj
                bij = a+aj
                s = math.sqrt(2*math.sqrt(aij)/bij) * math.exp(-aij/bij*dx2)
                S[i,j] = S[j,i] = s
                r = fR * math.exp(-a*dx2)
                if (abs(r) < 1e-99): r = 0
                R[i,j] = R[j,i] = r
                d = fD * dx
                g = (d*d + fD)*r
                d *= r
                if (abs(d) < 1e-99): d = 0
                # DS: a bug was here, original is:
                # D[i,j] = d
                # D[j,i] = -d
                # DS: now I changed to:
                D[i,j] = -d
                D[j,i] = d
                if (abs(g) < 1e-99): g = 0
                G[i,j] = G[j,i] = g


        Ri = LA.inverse(R)

        # Use constant mass or mass vector to calculate G and transform to
        # GRm = G * R^-1
        GRm = N.dot(N.multiply(-0.5/mass, G.T).T, Ri)
        GRm = 0.5 * ( GRm + N.transpose(GRm) )    # symmetrize

        self.GRm = GRm
        self.DRi = N.dot(D, Ri)
        self.GRi = N.dot(G, Ri)

        if keepMatrices:
            self.S = S
            self.R = R
            self.Ri = Ri
            self.A = A
            self.D = D
            self.G = G
            self.D1 = D
            self.D2 = G

    def __call__(self, V, mass = None, Ugrad = None, clear = False):
        if mass is None:
            # kinetic energy is constant and is stored in GRm
            try:
                self.H[:,:] = self.GRm
            except:
                self.H = copy(self.GRm)
        else:
            # recreate kinetic part of H with effective reduced mass
            self.H = N.multiply(-0.5/mass, self.GRi.T).T
            # and effective potential Ugrad for the gradient of psi
            if Ugrad is not None:
                self.H += N.multiply(Ugrad, self.DRi.T).T
            self.H = 0.5 * ( self.H + N.transpose(self.H) )  # symmetrize

        if len(V.shape) == 1 and len(V) == self.n:
            for i in range(self.n): self.H[i,i] += V[i]
        elif self.H.shape == V.shape: self.H += V
        else: raise IndexError
        Eigensolver.__call__(self, sym = True, overwrite = True)
        if clear: del self.H
#       N.multiply(self.ev, self.Rd, self.ev)
        # self.ev = N.dot(self.R, self.ev) # DS: removed, has no effect
        self.normalize()
        return self

    def normalize(self):
        """
        phase convention and normalization (Simpson's rule)
        """
        for i in range(self.nstates):
            v = self.Psi(i)
            area = N.sum(v[:self.n/2])
            nrm = 1.
            if i%2 == 0:
                if area < 0: nrm = -1.
            else:
                if area > 0: nrm = -1.
            v2 = v * v
            v2 = N.multiply(v2, self.weights, v2)
            nrm /= math.sqrt(N.sum(v2))
            v *= nrm

Collocate = GaussianCollocation1D

class CMDVR(Eigensolver):
    """
    Solving the vibrational Schrödinger equation with potential V using the
    Colbert Miller DVR method, JCP, 96(3), p-1982, (1992)
    Kinetic energy elements are obtained from Eqs. A6 of Appendix A.
    simple generic DVR, Uniform grid Fourier basis
    """

    def __init__(self, x, w = None,  nstates = None, mass = 1):
        n = len(x)
        if nstates is None: nstates = n
        self.grid = x
        if w is not None:
            self.weights = w
        
        else:       # assume Simpson's rule
            self.weights = N.zeros(len(x), nxFloat)
            for i in range(1,len(x)-1):
                self.weights[i] = x[i+1] - x[i-1]
            self.weights[0]  = x[1]  - x[0]
            self.weights[-1] = x[-1] - x[-2]
            self.weights *= 0.5
        
        self.n = n
        self.nstates = nstates
        self.fac1 = 0.25*pi*pi/(mass*(x[-1] - x[0])**2)
        self.fac2 = pi/(2*(self.n + 1))

    def __call__(self, V, clear=True):
        self.H = N.zeros((self.n,self.n),nxFloat)
        k = self.n + 1
        T = ((self.fac1 * ((2 * k**2 + 1)/3 - \
             1./N.power(N.sin(pi * N.arange(1,self.n+1).astype(nxFloat)/k),2)))+V)
        for i in range(self.n):
            self.H[i,i] = T[i]
            for j in range(i):
                self.H[i,j] = \
                self.fac1* (-1)**(i-j) * (1./N.power(N.sin(self.fac2*(i-j)),2) \
                 - (1./N.power(N.sin(self.fac2*(i+j+2)),2)))
#               self.H[j,i] = self.H[i,j]
        del T
        Eigensolver.__call__(self, sym=True)
        if clear: del self.H
        self.normalize()
        return self
    
    def normalize(self):
        """
        phase convention and normalization
        """
        for i in range(self.nstates):
            v = self.Psi(i)
            area = N.sum(v[:self.n/2])
            nrm = 1.
            if i%2 == 0:
                if area < 0: nrm = -1.
            else:
                if area > 0: nrm = -1.
            v2 = v * v
            v2 = N.multiply(v2, self.weights, v2)
            nrm /= math.sqrt(N.sum(v2))
            v *= nrm

class periodicDVR(Eigensolver):
    def __init__(self, x, w = None, nstates = None, mass = 1):
        """
        'mass' may be a scalar (constant mass) or a vector (representation of 
        a non-constant reduced mass on the grid).
        """
        n = len(x)
        if nstates is None: nstates = n

        self.grid = x
        T = x.period # period
        assert w is not None
        self.weights = w
        self.n = n
        self.nstates = nstates

        D1 = N.zeros((n, n))
        D2 = N.zeros((n, n))
        c_diag = (1 - n*n)*N.pi*N.pi/(3.0*T*T)
        c_offDiag = 2.*N.pi*N.pi/(T*T)
        # diagonal elements
        for i in range(1,n+1):
            D2[i-1,i-1] = c_diag
        # off-diagonal elements
        for i in range(1,n+1):
            for j in range(1, i):
                D1[i-1,j-1] = (
                        (N.pi/T)*(-1)**N.abs(j - i)/N.sin(N.pi*(i - j)/float(n))
                        )
            for j in range(i+1, n+1):
                D1[i-1,j-1] = (
                        (N.pi/T)*(-1)**N.abs(j - i)/N.sin(N.pi*(i - j)/float(n))
                        )
                D2[i-1,j-1] = D2[j-1,i-1] = (
                        (c_offDiag*(-1)**N.abs(1 + i - j)*
                        N.cos(((i - j)*N.pi)/n))/(N.sin(((i - j)*N.pi)/n)**2)
                        )
 
        self.D1 = D1
        self.D2 = D2
        self.DRi = D1
        self.GRi = D2

    def __call__(self, V, mass = None, Ugrad = None, clear = False):
        if mass is None:
            # kinetic energy is constant and is stored in D2
            try:
                self.H[:,:] = -0.5*self.D2
            except:
                self.H = copy(-0.5*self.D2)
        else:
            # recreate kinetic part of H with effective reduced mass
            self.H = N.multiply(-0.5/mass, self.D2.T).T
            # and effective potential Ugrad for the gradient of psi
            if Ugrad is not None:
                self.H += N.multiply(Ugrad, self.D1.T).T
            self.H = 0.5 * ( self.H + N.transpose(self.H) )  # symmetrize

        if len(V.shape) == 1 and len(V) == self.n:
            for i in range(self.n): self.H[i,i] += V[i]
        elif self.H.shape == V.shape: self.H += V
        else: raise IndexError
        Eigensolver.__call__(self, sym = True, overwrite = True)
        if clear: del self.H
        self.normalize()
        return self

    def normalize(self):
        """
        phase convention and normalization (Simpson's rule)
        """
        for i in range(self.nstates):
            v = self.Psi(i)
            area = N.sum(v[:self.n/2])
            nrm = 1.
            if i%2 == 0:
                if area < 0: nrm = -1.
            else:
                if area > 0: nrm = -1.
            v2 = v * v
            v2 = N.multiply(v2, self.weights, v2)
            nrm /= math.sqrt(N.sum(v2))
            v *= nrm


if __name__ == '__main__':
    
    def Morse(r, D = 5.716, a = 0.1519399, re = 0):
        V = (-a) *(N.array(r).astype(nxFloat) - re)
        V = N.exp(V, V)
        V = D * V * (V - 2)
        return V

    def tst1():
        n = 40
        x0 = -5
        xn = 20.
        x = N.arange(n+1)*(xn - x0)/n + x0
        V = Morse(x)
        c = Collocate(x)
        c(V)
        return x, V, c
    x,V,c = tst1()

