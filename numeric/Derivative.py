# thctk.numeric.Derivative
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
    This Module provides functions for evaluating numerical derivatives.
"""

import math
from thctk.numeric import *

def zero(x):
    return 0

def stepsize(xc, eta = 1.e-8, pow = 1./3):
    """
    h = stepsize(xc, eta = 1.e-8, pow = 1./3)

    returns the optimal stepsize for numeric derivatives where:

        xc  ... is the characteristic scale of variation of the function
                e.g. sqrt(f/f")
        eta ... is the relative error in evaluating the function
        pow ... fractional power

    Ref.: Numerical Recipes in Fortran, second edition, chapter 5.7
    """

    return xc * math.pow(eta, pow)

class Derivative:

    def __init__(self, f = zero, order = (1,)):
        self.order = order
        self.setf(f, len(order))

    def setf(self, f, dimension):
        self.function = f
        self.dimension = dimension
        if len(self.order) != self.dimension:
            self.order = (0,) * self.dimension

    def __call__(self, x):
        pass

def weights(xi, x, m):
    """
    c = weights(xi, x, m)

    weights for finite difference approximations of derivatives
    Ref: B. Fornberg, D.M. Sloan, Acta Numerica (1994), 203-267
         B. Fornberg, A practical guide to Pseudospectral Methods,
         Cambridge UP 1996, App. C

    xi is the point at which the derivatives are evaluated
    x contains the grid points around xi
        the grid points in x should be ordered so that |x_i| <= |x_{i+1}|,
        otherwise the weights for higher derivatives will have large errors
    m is the highest order derivative for which weights are computed

    the weights for the n'th derivative (n=0...m) that use all possible grid
    points are contained in c[:,-1,n] in the output array c. If the requested
    derivative can not be represented on the number of grid points given, all
    weights will be 0.
    """

    n = len(x)
    c = N.zeros([n,n,m+1], nxFloat)
    c[0,0,0] = 1.
    c1 = 1.
    c4 = x[0] - xi
    for i in range(1,n):
        mn = min(i,m) + 1
        c2 = 1.
        c5 = c4
        c4 = x[i] - xi
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2*c3
            if (i <= m): c[j,i-1,i] = 0.
            c[j,i,0] = c4*c[j,i-1,0]/c3
            for k in range(1,mn):
                c[j,i,k] = (c4*c[j,i-1,k] - k*c[j,i-1,k-1])/c3
        c[i,i,0] = -c1*c5*c[i-1,i-1,0]/c2
        for k in range(1,mn):
            c[i,i,k] = c1*(k*c[i-1,i-1,k-1] - c5*c[i-1,i-1,k])/c2
        c1 = c2
    return c

def get_x(xi, typx = 0, eta = 0, n = 4):
    """
    x = get_x(xi, typx = 0, eta = 0, n = 4)

    generate a grid for centered half-way point approximation
    xi is the point at which the derivatives are evaluated,
    typx is a typical scale for the variable (e.g. sqrt(f/f'')),
    eta is the relative noise in evaluating the function, and
    n is the number of grid points (n should be even)
    """
    x = []
    if eta == 0: eta = macheps()
    h = eta**(1./3.)*max(abs(xi), typx)
    temp = xi + h/2
    hh = temp - xi
    h = hh + hh
    a = xi - hh
    b = xi + hh
    x.append(a)
    x.append(b)
    for i in range(n/2-1):
        a = a - h
        b = b + h
        x.append(a)
        x.append(b)
    return x


def loadNetCDF(file='grid.nc'):
    """
    xi, g, w, f = loadNetCDF(file='grid.nc')
    """
    n = Scientific.IO.NetCDF.NetCDFFile(file, 'r')
    xi = n.variables['xi'].getValue()
    d = len(xi)
    g = []
    w = []
    for i in range(d):
        g.append(n.variables['g'+`i`].getValue())
        w.append(n.variables['w'+`i`].getValue())
    f = n.variables['f'].getValue()
    n.close()
    return xi, g, w, f

def saveNetCDF(xi, g, w, f, file='grid.nc'):
    """
    saveNetCDF(xi, g, w, f, file='grid.nc')
    """
    d = len(xi)
    fdims = []
    ng = []
    nw = []
    n = Scientific.IO.NetCDF.NetCDFFile(file, 'w')
    n.createDimension('ndim', d)
    for i in range(d):
        n.createDimension('n' + `i`, len(g[i]))
        n.createDimension('d' + `i`, len(w[i]))
        fdims.append('n' + `i`)
    n.sync()
    nx = n.createVariable('xi', ncFloat, ('ndim',))
    for i in range(d):
        np = 'n' + `i`
        nd = 'd' + `i`
        ng.append(n.createVariable('g' + `i`, ncFloat, (np,)))
        nw.append(n.createVariable('w' + `i`, ncFloat, (nd, np)))
    nf = n.createVariable('f', ncFloat, tuple(fdims))
    n.sync()

    nx.assignValue(xi)
    for i in range(d):
        ng[i].assignValue(g[i])
        nw[i].assignValue(w[i])
    n.sync()
    nf.assignValue(f)
    n.sync()

    n.close()

def derivative(w, f, d):
    """
    s = derivative(w, f, d)

    calculate the derivative of the function f given on a grid with the
    corresponding weights w
    the orders of the derivative in all dimensions are given in d
    """
    Na = N.sum
    NA = NewAxis
    s = 'f'
    n = len(d)
    for i in range(n-1):
        j = n - i - 1
        s = 'Na(' + s + '*w[' + `i` + '][' + `d[i]` + '][:' + ',NA'*j + '])'
    s = 'Na(' + s + '*w[' + `n-1` + '][' + `d[n-1]` + '])'
#   return s
    return eval(s)

def weightsN(xi, grid, m = 3):
    """
    w = weightsN(xi, grid, m = 3)

    compute the finite difference weights for derivatives up to order m on an
    N-dimensional grid centered at xi
    the weights returned in w are ordered: w[dimension][order, grid point]
    """
    n = len(xi)
    w = []
    for i in range(n):
        w.append(N.transpose(weights(xi[i], grid[i], m)[:,-1,:]))
    return w

def fgrid(fn, xi, typx = 0.01, eta = 0):
    """
    grid, f = fgrid(fn, xi, typx = 0.01, eta = 0)

    evaluate the function fn on a grid for determining its numerical
    derivatives; the grid is obtained by calls to get_x()

    xi is the point at which the derivatives are calculated
    typx contains the typical scales of the dimensions in xi
    eta is the precision in evaluating fn
    """
    d = len(xi)
    if type(eta) == IntType or type(eta) == FloatType:
        if eta == 0: eta = macheps()
        eta = eta*N.ones(d, nxFloat)
    if type(typx) == IntType or type(typx) == FloatType:
        typx = typx*N.ones(d, nxFloat)
    grid = []
    shape = []
    for i in range(d):
        g = get_x(xi[i], typx[i], eta[i], 4)
        shape.append(len(g))
        grid.append(g)
    f = N.zeros(shape, nxFloat)
    subgrid(fn, grid, f)
    return grid, f


def subgrid(fn, grid, f, x = [], i = []):
    """
    subgrid(fn, grid, f, x = [], i = [])

    recursively evaluate the function fn on a grid

    grid is a list of lists containing the grid points in the dimensions
        1...len(grid)
    fn is a function of an len(grid)-dimensional vector
    x are the first d components of a vector on the grid
    i is the corresponding index vector
    f is the resulting array of function values on the grid with
        shape = [len(grid[0]), len(grid[1]), ... ]
    """
    d = len(x)
    if d == len(grid):
        f[i] = fn(x)
    else:
        for k in range(len(grid[d])):
            x1 = x + [grid[d][k],]
            i1 = i + [k,]
            subgrid(fn, grid, f, x1, i1)

