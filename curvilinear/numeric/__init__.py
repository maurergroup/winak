# thctk.numeric initialization
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
from INTERNALS.curvilinear.numeric._numeric import numericBackend
numericBackendNumeric, numericBackendNumpy = 0, 1

# FFT is used rarely and handled in each file depending on numericBackend
# see e.g.: spectroscopy/Exciton.py

def importLinearAlgebra():
    if numericBackend == 0:
        import LinearAlgebra as LA
    elif numericBackend == 1:
        import numpy.oldnumeric.linear_algebra as LA
    else:
        raise ImportError('no valid LinearAlgebra package specified!')
    return LA

if numericBackend == numericBackendNumeric:
    import Numeric as N
    from UserArray import UserArray
    NewAxis = N.NewAxis
    nxArrayType = N.ArrayType
    nxFloat = N.Float
    nxFloat0 = N.Float0
    nxFloat32 = N.Float32
    nxFloat64 = N.Float64
    nxInteger = N.Int
    nxInt = N.Int
    nxInt8 = N.Int8
    nxInt16 = N.Int16
    nxInt32 = N.Int32
    nxUnsignedInt8 = N.UnsignedInt8
    nxCharacter = N.Character
    nxComplex = N.Complex
    ncFloat = N.Float
    ncFloat0 = N.Float0
    ncFloat32 = N.Float32
    ncFloat64 = N.Float64
    ncInteger = N.Int
    ncInt = N.Int
    ncInt8 = N.Int8
    ncInt16 = N.Int16
    ncInt32 = N.Int32
    ncUnsignedInt8 = N.UnsignedInt8
    ncCharacter = N.Character
    nxTypecode = lambda obj: getattr(obj, 'typecode')()
    N.nnzSearch = N.nonzero
elif numericBackend == numericBackendNumpy:
    import numpy as N
    import numpy.core.multiarray, numpy.oldnumeric
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
    nxUnsignedInt8 = numpy.oldnumeric.UnsignedInt8
    nxCharacter = N.character
    nxComplex = N.complex
    nxComplex64 = N.complex64
    nxComplex128 = N.complex128
    ncFloat = numpy.oldnumeric.Float
    ncFloat0 = numpy.oldnumeric.Float0
    ncFloat32 = numpy.oldnumeric.Float32
    ncFloat64 = numpy.oldnumeric.Float64
    ncInteger = numpy.oldnumeric.Int
    ncInt = numpy.oldnumeric.Int
    ncInt8 = numpy.oldnumeric.Int8
    ncInt16 = numpy.oldnumeric.Int16
    ncInt32 = numpy.oldnumeric.Int32
    ncUnsignedInt8 = numpy.oldnumeric.UnsignedInt8
    ncCharacter = numpy.oldnumeric.Character
    nxTypecode = lambda obj: getattr(obj, 'dtype')
    N.nnzSearch = numpy.oldnumeric.nonzero
else:
    raise ImportError('no valid numeric backend specified!')
