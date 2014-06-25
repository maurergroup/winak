# thctk.numeric.IMLS
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2006 Christoph Scheurer
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
Interpolating Moving Least Squares Interpolation
"""

import types, math
from thctk.numeric import *
from scipy.linalg.fblas import dnrm2
from thctk.numeric.gelsy import dgelsy

class cartesianMetric2:
    """
    for the call we always assume that the first argument might contain
    multiple vectors while the second argument will only be a single vector
    """

    def __init__(self):
        pass

    def __call__(self, x, x0):
        v = x - x0
        v *= v
        d = N.sum(v)
        return N.sqrt(d, d)

    def grad(self, x, x0):
        pass

class GaussWeight:

    def __init__(self, alpha = 1.0, eps = 0.0, np = 2, inverseDist = False):
        self.alpha = alpha
        self.eps = eps
        self.np = np
        self.inverseDist = inverseDist

    def __call__(self, d):
        dd = N.array(d)
        if self.inverseDist:
            dd = 1/dd
        dn = N.power(dd, self.np)
        dn += self.eps
        dd *= dd
        dd *= -self.alpha
        dd = N.exp(dd, dd)
        dd /= dn
        return dd

    def D(self, d):
        if self.inverseDist:
            d = 1/d
        dd = N.array(d*d)
        dd *= -self.alpha
        dd = N.exp(dd, dd)
        dn1 = N.power(d, self.np-1)
        dn = dn1*d
        dn += self.eps
        dd /= dn
        dd /= dn
        dn *= d
        dn *= -2*self.alpha
        dn1 *= self.np
        dn -= dn1
        dd *= dn
        if self.inverseDist:
            dd *= -d*d
        return dd

class Bmatrix:

    def __init__(self, Z):
        self.Z = N.array(Z).astype(nxFloat)
        self.Zt = N.transpose(self.Z)
        m, d = self.Z.shape     # m points with d coordinates
        self.m = m
        self.d = d
        self.tmpd = N.zeros(self.d, nxFloat)
        self.tmpm = N.zeros(self.m, nxFloat)

    def __call__(self, x):
        """evaluate the basis functions at the point x"""
        return None

    def dx(self, x, s):
        """evaluate the first derivative of the basis functions w.r.t.
        variable s at the point x"""
        return None

    def dx2(self, x, s, t):
        """evaluate the second derivative of the basis functions w.r.t.
        variables s and t at the point x"""
        return None

    def Matrix(self, B = None):
        """return full matrix representation of B"""
        return None

    def Mv(self, v, res = None):
        """Matrix times vector v product"""
        return None

    def Tv(self, v, res = None):
        """Transpose times vector v product"""
        return None

class BmatrixBasisFunctions(Bmatrix):

    def __init__(self, Z, functions):
        Bmatrix.__init__(self, Z)
        n = len(functions) + 1  # we always add the constant function
        self.functions = functions
        self.n = n
        self.shape = (n, self.m)
        self.tmpn = N.zeros(self.n, nxFloat)

    def __call__(self, x, res = None):
        if res is None: res = N.zeros(self.n, nxFloat)
        res[0] = 1
        k = 1
        for f in self.functions:
            res[k] = f(x)
            k += 1
        return res

    def Matrix(self, B = None):
        if B is None: B = N.zeros(self.shape, nxFloat)
        self.B = B
        k = 0
        for x in self.Z:
            res = self(x, B[:,k])
            k += 1
        return B

class BmatrixPolynomial2(Bmatrix):

    def __init__(self, Z):
        Bmatrix.__init__(self, Z)
# number of basis functions: constant + linear + quadratic terms
        n = 1 + self.d + (self.d*(self.d+1))/2
        self.n = n
        self.shape = (n, self.m)
        self.tmpn = N.zeros(self.n, nxFloat)

    def __call__(self, x, res = None):
        if res is None: res = N.zeros(self.n, nxFloat)
        d = self.d
        res[0] = 1
        k = d + 1
        res[1:k] = x
        for i in range(d):
            for j in range(i, d):
                res[k] = x[i]*x[j]
                k += 1
        return res

    def dx(self, x, s, res = None):
        """evaluate the first derivative of the basis functions w.r.t.
        variable s at the point x"""
        d = self.d
        if s < 0 or s >= d:
            raise IndexError
        if res is None:
            res = N.zeros(self.n, nxFloat)
        else:
            res[:] = 0
        k = 1 + s
        res[k] = 1
        for i in range(s):
            k += d
            res[k] = x[i]
        k += d - s
        res[k] = 2*x[s]
        for i in range(s+1,d):
            k += 1
            res[k] = x[i]
        return res

    def dx2(self, x, s, t, res = None):
        """evaluate the second derivative of the basis functions w.r.t.
        variables s and t at the point x"""
        d = self.d
        u = min(s, t)
        v = max(s, t)
        if u < 0 or v >= d:
            raise IndexError
        if res is None:
            res = N.zeros(self.n, nxFloat)
        else:
            res[:] = 0
        k = 1 + d + (v - u)
        k += u*d - ((u-1)*u)/2
        if u == v:
            res[k] = 2
        else:
            res[k] = 1
        return res

    def Matrix(self, B = None):
        d = self.d
        Z = self.Zt
        if B is None: B = N.zeros(self.shape, nxFloat)
        self.B = B
        B[0] = 1
        k = d + 1
        B[1:k] = Z
        for i in range(d):
            for j in range(i, d):
                N.multiply(Z[i], Z[j], B[k])
                k += 1
        return B

    def Mv(self, v, res = None):
        if res is None: res = N.zeros(self.n, nxFloat)
        Z = self.Zt
        t = self.tmpm
        d = self.d
        k = 0
        res[k] = N.sum(v)
        for i in range(d):
            k += 1
            res[k] = N.dot(Z[i], v)
        for i in range(d):
            t = N.multiply(v, Z[i], t)
            for j in range(i, d):
                k += 1
                res[k] = N.dot(Z[j], t)
        return res

    def Tv(self, v, res = None):
        if res is None: res = N.zeros(self.m, nxFloat)
        Zt = self.Zt
        t = self.tmpm
        d = self.d
        res = v[0]
        k = d + 1
        res += N.dot(self.Z, v[1:k])
        for i in range(d):
            for j in range(i, d):
                N.multiply(Zt[i], Zt[j], t)
                t *= v[k]
                res += t
                k += 1
        return res

class interpolate:

    def __init__(self, nodes = [], values = [], weight = None, B = None,
            rcond = 1e-12):
        """
        nodes is a list of vectors at the positions where values are given
        weight is the IMLS weight function
        B is a Bmatrix object, by default a BmatrixPolynomial2
        """
        self.nodes = nodes
        self.values = values
        self.rcond = rcond
        if len(nodes):
            self.dim = len(nodes[0])
        else:
            self.dim = 0
        if weight is None:
            weight = GaussWeight(alpha = 1.5, eps = 1e-6)
        self.weight = weight
        if B is None:
            B = BmatrixPolynomial2(self.nodes)
        self.B = B
        self.a = N.zeros(max(self.B.shape), nxFloat)
        self.W = N.zeros(len(self), nxFloat)
        self.W2 = N.zeros(len(self), nxFloat) # sqrt(W)
        self.x = None
        self.tmpv = N.zeros(self.dim, nxFloat)

    def __call__(self, x):
        """
        evaluate the interpolation at position x
        """
        self.x = N.array(x).astype(nxFloat)
        self.setupW()
        return self.interpolate()

    def grad(self, x = None, grad = None, maxsize = None):
        generalPoint = (type(x) != types.IntType)
        if generalPoint:
            if x is not None:
                raise ValueError("x can only be a grid point index")
            else:
                x = self.x
        else:
            x = N.array(self.nodes[x])
        if grad is None:
            grad = N.zeros(self.dim, nxFloat)
        tmp = N.zeros(self.B.m, nxFloat)
        a = self.a[:self.B.m]
        for i in xrange(self.dim):
            grad[i] = N.dot(a, self.B.dx(x, i, res = tmp))
        del tmp
        if not generalPoint: # finished if x is one of the data points
            return grad
        # We have a general point and need to evaluate the derivatives of the
        # coefficients
        rhs = N.dot(N.transpose(self.Bmatrix), self.coeff)
        rhs -= self.values
        nrhs = self.setupWorkspace(nrhs = self.dim, maxsize = maxsize)
        if nrhs != len(grad):
            raise NameError, "ImplementationError"
        A = self.gradW(rhs)
        self.jpvt[:] = 0
        R = N.multiply(self.W2[:,NewAxis], N.transpose(self.Bmatrix), self.R)
        A, self.rank, self.info = \
            dgelsy(R, A, self.jpvt, self.rcond, self.work)
        m, n = self.B.shape
        for i in xrange(self.dim):
            grad[i] += N.dot(A[:m,i], self.Bx)
        return grad

    def gradW(self, rhs):
        np = len(self.distances)
        self.dW = self.weight.D(self.distances)
        self.dW /= self.distances
        self.dW /= self.W2
        self.dW *= rhs
        A = N.zeros((np, self.dim), nxFloat)
        for i in xrange(np):
            N.subtract(self.nodes[i], self.x, A[i]) # p - x
        A *= self.dW[:,NewAxis]
        return A

    def gradNode(self, n, grad = None):
        return self.grad(x = n, grad = grad)

    def setupDenseInterpolation(self):
        self.Bmatrix = self.B.Matrix()
        m, n = self.Bmatrix.shape
        self.R = N.zeros( (n,m), dtype = nxFloat, order = 'F')
        self.jpvt = N.zeros(m, nxInt)
        nrhs = self.setupWorkspace()

    def setupWorkspace(self, nrhs = 1, maxsize = None):
        work = N.zeros(2, nxFloat)
        for n in xrange(nrhs, 0, -1):
            a, rank, info = dgelsy(self.R, self.a, self.jpvt, self.rcond, \
                work, nrhs=n, lwork=-1)
            lwork = int(work[0])
            if maxsize is None or lwork <= maxsize:
                maxrhs = n
                break
        if hasattr(self, 'work') and len(self.work) >= lwork:
            pass
        else:
            self.work = N.zeros(lwork, nxFloat)
        return maxrhs

    def interpolate(self):
        """
        interpolate using the full matrix
        """
        if not hasattr(self, 'Bmatrix'):
            self.setupDenseInterpolation()
        B = self.Bmatrix
        m, n = B.shape
        self.jpvt[:] = 0
        a = N.multiply(self.W2, N.array(self.values), self.a[:n])
        R = N.multiply(self.W2[:,NewAxis], N.transpose(B), self.R)
        a, self.rank, self.info = \
            dgelsy(R, self.a, self.jpvt, self.rcond, self.work, nrhs=1)
        self.Bx = self.B(self.x)
        self.coeff = a[:self.Bx.shape[0]]
        return N.dot(self.coeff, self.Bx)

    def dist(self, a, b):
        """
        this is the external interface to the distance calculation used in
        setupW()
        """
        d = N.array(a) - N.array(b)
        return N.sqrt(N.dot(d, d))

    def setupW(self, x = None, distances = None):
        """
        build the vector of weights w(d(x, x_i))
        """
        if x is None:
            x = self.x
        else:
            x = N.array(x).astype(nxFloat)
        if x is None:
            raise ValueError("x is None")
        if distances is None:
            self.distances = N.zeros(len(self.nodes), nxFloat)
            t = self.tmpv
            for i in range(len(self)):
                self.distances[i] = dnrm2(N.subtract(x, self.nodes[i], t))
        else:
            self.distances = N.array(distances)
        self.W  = self.weight(self.distances)
        self.W2 = N.sqrt(self.W, self.W2)

    def __len__(self):
        return len(self.values)

class cutoffWeight:

    def __init__(self, weight, nfrac = 0.1, nmin = 5, cutoffType = 'cos'):
        self.weight = weight
        self.nfrac = nfrac
        self.nmin = nmin
        self.cutoffType = cutoffType

    def __call__(self, d):
        d = N.array(d)
        w = self.weight(d)
        nmin = max(self.nmin, 2)
        n = max(nmin, int(len(d)*self.nfrac))
        if self.cutoffType == 'cos':
            c = d[-n:].copy()
            c -= c[0]
            c *= math.pi/c[-1]
            c = N.cos(c, c)
            c += 1
            c *= 0.5
        else:
            raise NotImplementedError
        w[-n:] *= c
        return w

class cutoffInterpolation:

    def __init__(self, nodes = [], values = [], weight = None, B = None,
            rcond = 1e-12, nmax = 100):
        self.nodes = N.array(nodes)
        self.values = values
        self.rcond = rcond
        if len(nodes):
            self.dim = len(nodes[0])
        else:
            self.dim = 0
        if weight is None:
            weight = GaussWeight(alpha = 1.5, eps = 1e-6)
        self.weight = weight
        self.cutoffWeight = cutoffWeight(self.weight)
        self.B = B
        self.nmax = nmax
        self.tmpA = N.zeros(self.nodes.shape, nxFloat)

    def __call__(self, x):
        """
        evaluate the interpolation at position x
        """
        self.x = N.array(x).astype(nxFloat)
        d = self.distance(x)
        order = N.argsort(d)
        useNodes = order[:self.nmax]
        self.useNodes = useNodes
        # nodes = N.take(self.nodes, useNodes)
        # 'N.take' changed, is no axis is specified, the array is treated flat.
        nodes = N.take(self.nodes, useNodes, axis = 0)
        values = N.take(self.values, useNodes)
        distances = N.take(d, useNodes)
        self.IMLS = interpolate(nodes, values, weight = self.cutoffWeight,
            B = self.B, rcond = self.rcond)
        self.IMLS.x  = self.x
#       self.IMLS.setupW(distances = distances)
        self.IMLS.setupW()
        return self.IMLS.interpolate()

    def distance(self, x):
        self.tmpA = N.subtract(self.nodes, x, self.tmpA)
        self.tmpA *= self.tmpA
        return N.sqrt(N.sum(self.tmpA, 1))
