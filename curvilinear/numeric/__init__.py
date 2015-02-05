# numeric initialization
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
"""
from winak.curvilinear.numeric._numeric import numericBackend
numericBackendNumpy = 1

# FFT is used rarely and handled in each file depending on numericBackend
# see e.g.: spectroscopy/Exciton.py

if numericBackend == numericBackendNumpy:
    import numpy as N
    import numpy.core.multiarray
    from numpy.lib.user_array import container as UserArray
    NewAxis = None
    nxArrayType = numpy.core.multiarray.ndarray
    nxFloat = N.float
    nxFloat0 = N.float32
    nxFloat32 = N.float32
    nxFloat64 = N.float64
    nxInteger = N.integer
    nxInt = N.int
    nxInt8 = N.int8
    nxInt16 = N.int16
    nxInt32 = N.int32
    nxInt64 = N.int64
    nxCharacter = N.character
    nxComplex = N.complex
    nxComplex64 = N.complex64
    nxComplex128 = N.complex128
    nxTypecode = lambda obj: getattr(obj, 'dtype')
else:
    raise ImportError('no valid numeric backend specified!')
