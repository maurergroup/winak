# thctk.numeric.Spline
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
    This Module provides spline interpolations
"""

from thctk.numeric import *
import math
from thctk.numeric.bspline_22 import *

class Spline:

    def __init__(self):
        pass

class Spline1D(Spline):

    def __init__(self, x, data, kx = 3):
        self.x = N.array(x).astype(nxFloat)
        self.kx = kx
        self.xknot = dbsnak(self.x, self.kx)
        self.data = N.array(data).astype(nxFloat)
        self.shape = self.data.shape
        self.xrange = (min(self.x), max(self.x))
        self.coeff = dbsint(self.x, self.data, self.kx, self.xknot)

    def __call__(self, x):
        return dbsval(x, self.kx, self.xknot, self.coeff)

    def grid(self, x):
        n = len(x)
        f = N.zeros(n, nxFloat)
        for i in range(n): f[i] = self(x[i])
        return f

    def D(self, i, x):
        return dbsder(i, x, self.kx, self.xknot, self.coeff)

class Spline2D(Spline):

    def __init__(self, x, y, data, kx = 3, ky = 3):
        self.x = N.array(x).astype(nxFloat)
        self.y = N.array(y).astype(nxFloat)
        self.kx = kx
        self.ky = ky
        self.xknot = dbsnak(self.x, self.kx)
        self.yknot = dbsnak(self.y, self.ky)
        self.data = N.array(data).astype(nxFloat)
        self.shape = self.data.shape
        self.xrange = (min(self.x), max(self.x))
        self.yrange = (min(self.y), max(self.y))
        self.coeff = dbs2in(self.x, self.y, self.data, self.kx, self.ky,
            self.xknot, self.yknot)

    def __call__(self, x, y):
        return dbs2vl(x, y, self.kx, self.ky, self.xknot, self.yknot, self.coeff)

    def grid(self, x, y):
        nx = len(x)
        ny = len(y)
        f = N.zeros((nx, ny), nxFloat)
        for i in range(nx):
            for j in range(ny): f[i,j] = self(x[i], y[j])
        return f

    def D(self, i, j, x, y):
        return dbs2dr(i, j, x, y, self.kx, self.ky, self.xknot, self.yknot,
            self.coeff)

class Spline3D(Spline):

    def __init__(self, x, y, z, data, kx = 3, ky = 3, kz = 3):
        self.x = N.array(x).astype(nxFloat)
        self.y = N.array(y).astype(nxFloat)
        self.z = N.array(z).astype(nxFloat)
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.xknot = dbsnak(self.x, self.kx)
        self.yknot = dbsnak(self.y, self.ky)
        self.zknot = dbsnak(self.z, self.kz)
        self.data = N.array(data).astype(nxFloat)
        self.shape = self.data.shape
        self.xrange = (min(self.x), max(self.x))
        self.yrange = (min(self.y), max(self.y))
        self.zrange = (min(self.z), max(self.z))
        self.coeff = dbs3in(self.x, self.y, self.z, self.data, self.kx,
            self.ky, self.kz, self.xknot, self.yknot, self.zknot)

    def __call__(self, x, y, z):
        return dbs3vl(x, y, z, self.kx, self.ky, self.kz, self.xknot,
            self.yknot, self.zknot, self.coeff)

    def grid(self, x, y, z):
        nx = len(x)
        ny = len(y)
        nz = len(z)
        f = N.zeros((nx, ny, nz), nxFloat)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz): f[i,j,k] = self(x[i], y[j], z[k])
        return f

    def D(self, i, j, k, x, y, z):
        return dbs3dr(i, j, k, x, y, z, self.kx, self.ky, self.kz, self.xknot,
            self.yknot, self.zknot, self.coeff)

SplineNDclass = [None, Spline1D, Spline2D, Spline3D]
