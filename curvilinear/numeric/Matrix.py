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
from winak.curvilinear.numeric import *
from scipy.linalg import svd as SVD
from numpy.linalg import pinv

def regularizedInverse(A, eps = 1e-15):
    ##V, S, WT = N.linalg.svd(A, full_matrices = 0)
    # V, S, WT = SVD(A)
    # S2 = S*S
    # S2 += eps*eps
    # S /= S2
    # Ainv = N.dot(V, S[:,NewAxis]*WT)
    # return Ainv

    return pinv(A,rcond=eps)
