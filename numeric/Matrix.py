# thctk.numeric.Matrix
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
    This Module 
"""

import math
from thctk.numeric import *
LA = importLinearAlgebra()

SVD = LA.singular_value_decomposition

def directProduct(*matrices):
    if len(matrices) == 0: return 0
    elif len(matrices) == 1: return matrices[0]
    m = 1
    n = 1
    b = N.ones((1,))
    s1 = []
    s2 = []
    for k in range(len(matrices)):
        a = matrices[k]
        s1.append(m)
        s2.append(n)
        m = m * a.shape[0]
        n = n * a.shape[1]
        b = b * a[0,0]
    s1.reverse()
    s2.reverse()
    p = N.ones((m,n), nxTypecode(b))
    for k in range(len(matrices)):
        a = matrices[k]
        di = s1[k]
        dj = s2[k]
        ni = a.shape[0]
        nj = a.shape[1]
        for li in range(0,m,di*ni):
            for lj in range(0,n,dj*nj):
                for i in range(ni):
                    for j in range(nj):
                        aij = a[i,j]
                        for ri in (li+i*di)+N.arange(di):
                            for rj in (lj+j*dj)+N.arange(dj):
#                               print li, lj, ri, rj
                                p[ri,rj] = p[ri,rj] * aij

    return p

def regularizedInverse(A, eps = 1e-12):
    # V, S, WT = N.linalg.svd(A, full_matrices = 0)
    V, S, WT = SVD(A)
    S2 = S*S
    S2 += eps*eps
    S /= S2
    Ainv = N.dot(V, S[:,NewAxis]*WT)
    return Ainv

