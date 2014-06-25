# thctk.numeric.Polynomial
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2002 Christoph Scheurer
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
    This Module provides handling of polynomials in one and more dimensions
"""

import math
from thctk.numeric import *
from thctk.numeric._numeric import poly_eval, poly_terms
from copy import copy

def polcoe(x, y):
    """
    compute coefficients of the interpolating polynomial through (x_i, y_i)
    Numerical Recipes in Fortran, Ch. 3.5
    """

    n = len(x)
    s = N.zeros(n, nxFloat)
    c = N.zeros(n, nxFloat)
    s[-1] = -x[0]

    for i in range(1,n):
        for j in range(n-1-i,n-1):
            s[j] -= x[i]*s[j+1]
        s[-1] -= x[i]
    for j in range(n):
        phi = n
        for k in range(n-1,0,-1):
            phi = k*s[k] + x[j]*phi
        ff = y[j]/phi
        b = 1
        for k in range(n-1,-1,-1):
            c[k] += b*ff
            b = s[k] + x[j]*b
    return c

def neville(x, y, x0):
    """Through any N points y[i] = f(x[i]), there is a unique
    polynomial P order N-1.  Neville's algorithm is used for finding
    interpolates of this unique polynomial at any point x."""

    n = len(x)
    p = n*[0]
    for k in range(n):
        for j in range(n-k):
            if k == 0:
                p[j] = y[j]
	    else:
                p[j] = ((x0-x[j+k])*p[j]+(x[j]-x0)*p[j+1])/(x[j]-x[j+k])
    return p[0]

class Polynom:

    def __init__(self, dim, ncoeff = 0, typecode = nxFloat):
        self.dim = dim
        self.typecode = typecode
        if ncoeff:
            self.c = N.zeros(ncoeff, typecode)
            self.t = N.zeros((ncoeff, dim), nxInteger)
        else:
            self.c = None
            self.t = None

    def __call__(self, *x):
        t = self.typecode
        if len(x) == 1:
            x = N.array(x[0]).astype(t)
            if x.shape != (self.dim,): raise IndexError
        elif len(x) == self.dim: x = N.array(x).astype(t)
        else: raise TypeError
        return poly_eval(self.t, self.c, x)

    def __getitem__(self, index): return  self.c[index], self.t[index]

    def __len__(self): return self.t.shape[0]

    def __repr__(self): return `self.t`

    def __copy__(self):
        new = Polynom(self.dim, typecode = self.typecode)
        new.c = copy(self.c)
        new.t = copy(self.t)
        return new

    def terms(self, x, list = None, res = None):
        if list is None: list = N.arange(len(self))
        if res is None: res = N.zeros(len(self), self.typecode)
        return poly_terms(self.t, self.c, N.array(x).astype(self.typecode),
                          list, res)

    def nonzero(self, list = None):
        n = N.nonzero(self.c)
        if list is None: return n
        else: return n, N.take(N.array(list), n)

    def reduce(self, n = None):
        if n is None: n = self.nonzero()
        new = Polynom(self.dim, typecode = self.typecode)
        new.c = N.take(self.c, n)
        new.t = N.take(self.t, n)
        return new

    def coefficients(self, c):
        c = N.array(c).astype(self.typecode)
        if c.shape != (len(self),): raise IndexError
        self.c = c

    def D(self, *d):
        if len(d) != self.dim: raise IndexError
        new = copy(self)
        for i in range(new.dim):
            for j in range(d[i]): new.deriv(i)
        return new

    def deriv(self, i):
        if i < 0 or i > self.dim: raise IndexError
        for j in range(self.c.shape[0]):
            if self.c[j] != 0:
                if self.t[j,i]:
                    self.c[j] *= self.t[j,i]
                    self.t[j,i] -= 1
                else:
                    self.c[j] = 0
                    self.t[j] *= 0

    def totalDegree(self, n):
        dim = self.dim
        self.t = N.array(self.totdeg(n, dim - 1, [], 0, 0, N.zeros(dim)))
        self.c = N.zeros(len(self), self.typecode)

    def totdeg(self, n, d, p, c, k, a):
        for i in range(n-k+1):
            a[c] = i
            if c < d: p = self.totdeg(n,d,p,c+1,k+i,a)
            else: p.append(copy(a))
        return p

def binom(n, k):
    c = 1.
    for i in range(n-k+1,n+1): c *= i
    for i in range(1,k+1): c /= i
    return int(c + 0.5)
