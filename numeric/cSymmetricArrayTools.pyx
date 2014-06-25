import numpy as N
cimport numpy as N
cimport cython

N.import_array()

if sizeof(double) != sizeof(N.float_t):
    raise TypeError(
    "Size of 'double' (%i bit) is not equal to size of 'N.float_t' (%i bit)! "
                                %(sizeof(double), sizeof(N.float_t)) +
    "The cblas routines and the ndarray types won't fit."
    )


class dSymmetricArray2D(N.ndarray):
    def __setitem__(N.ndarray self, tuple ij, N.float_t value):
        cdef int t, i = ij[0], j = ij[1]
        if i < j: t = i ; i = j ; j = t
        i = i*(i+1)/2 + j
        assert i < self.shape[0]
        (<N.float_t*>self.data)[i] = value

    def __getitem__(N.ndarray self, tuple ij):
        cdef int t, i = ij[0], j = ij[1]
        if i < j: t = i ; i = j ; j = t
        i = i*(i+1)/2 + j
        assert i < self.shape[0]
        return (<N.float_t*>self.data)[i]

    def __repr__(self, ):
        return "dSymmetricArray2D(\n      " + (
                    N.ndarray.__repr__(N.asarray(self)))[6:]

class iSymmetricArray2D(N.ndarray):
    def __setitem__(N.ndarray self, tuple ij, N.int_t value):
        cdef int t, i = ij[0], j = ij[1]
        if i < j: t = i ; i = j ; j = t
        i = i*(i+1)/2 + j
        assert i < self.shape[0]
        (<N.int_t*>self.data)[i] = value

    def __getitem__(N.ndarray self, tuple ij):
        cdef int t, i = ij[0], j = ij[1]
        if i < j: t = i ; i = j ; j = t
        i = i*(i+1)/2 + j
        assert i < self.shape[0]
        return (<N.int_t*>self.data)[i]

    def __repr__(self, ):
        return "iSymmetricArray2D(\n      " + (
                    N.ndarray.__repr__(N.asarray(self)))[6:]


class SymmetricArray2D(N.ndarray):
    def __setitem__(N.ndarray self, tuple ij, value):
        cdef int t, i = ij[0], j = ij[1]
        if i < j: t = i ; i = j ; j = t
        # N.asarray(self)[i*(i+1)/2 + j] = value
        N.ndarray.__setitem__(self, i*(i+1)/2 + j, value) # faster

    def __getitem__(N.ndarray self, tuple ij):
        cdef int t, i = ij[0], j = ij[1]
        if i < j: t = i ; i = j ; j = t
        # return N.asarray(self)[i*(i+1)/2 + j]
        return N.ndarray.__getitem__(self, i*(i+1)/2 + j) # faster

    def __repr__(self, ):
        return "SymmetricArray2D(\n      " + (
                    N.ndarray.__repr__(N.asarray(self)))[6:]

class Cache1D(N.ndarray):
    def __setitem__(N.ndarray self, tuple index, N.float_t value):
        cdef int ni, mi, t
        (ni, mi) = index
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        (<N.float_t*>self.data)[t] = value

    def __getitem__(N.ndarray self, tuple index):
        cdef int ni, mi, t
        (ni, mi) = index
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        return (<N.float_t*>self.data)[t]

    def __repr__(self, ):
        return "Cache1D(\n      " + (
                    N.ndarray.__repr__(N.asarray(self)))[6:]

    def __str__(self, ):
        return repr(self)

class Cache2D(N.ndarray):
    def __setitem__(N.ndarray self, tuple index, N.float_t value):
        cdef int ni, mi, nj, mj, t, nMax = self.nMax
        (ni, mi, nj, mj) = index
        # i > j is implied here
        ni = ni*nMax + nj
        mi = mi*nMax + mj
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        (<N.float_t*>self.data)[t] = value

    def __getitem__(N.ndarray self, tuple index):
        cdef int ni, mi, nj, mj, t, nMax = self.nMax
        (ni, mi, nj, mj) = index
        # i > j is implied here
        ni = ni*nMax + nj
        mi = mi*nMax + mj
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        return (<N.float_t*>self.data)[t]

    def __repr__(self, ):
        return "Cache2D(\n      " + (
                    N.ndarray.__repr__(N.asarray(self)))[6:]

    def __str__(self, ):
        return repr(self)

class Cache3D(N.ndarray):
    def __setitem__(N.ndarray self, tuple index, N.float_t value):
        cdef int ni, mi, nj, mj, nk, mk, t, nMax = self.nMax
        (ni, mi, nj, mj, nk, mk) = index
        # i > j is implied here
        ni = ni*nMax*nMax + nj*nMax + nk
        mi = mi*nMax*nMax + mj*nMax + mk
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        (<N.float_t*>self.data)[t] = value

    def __getitem__(N.ndarray self, tuple index):
        cdef int ni, mi, nj, mj, nk, mk, t, nMax = self.nMax
        (ni, mi, nj, mj, nk, mk) = index
        # i > j is implied here
        ni = ni*nMax*nMax + nj*nMax + nk
        mi = mi*nMax*nMax + mj*nMax + mk
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        return (<N.float_t*>self.data)[t]

    def __repr__(self, ):
        return "Cache3D(\n      " + (
                    N.ndarray.__repr__(N.asarray(self)))[6:]

    def __str__(self, ):
        return repr(self)

class Cache4D(N.ndarray):
    def __setitem__(N.ndarray self, tuple index, N.float_t value):
        cdef int ni, mi, nj, mj, nk, mk, nl, ml, t, nMax = self.nMax
        (ni, mi, nj, mj, nk, mk, nl, ml) = index
        # i > j is implied here
        ni = ni*nMax*nMax*nMax + nj*nMax*nMax + nk*nMax + nl
        mi = mi*nMax*nMax*nMax + mj*nMax*nMax + nMax*mk + nl
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        (<N.float_t*>self.data)[t] = value

    def __getitem__(N.ndarray self, tuple index):
        cdef int ni, mi, nj, mj, nk, mk, nl, ml, t, nMax = self.nMax
        (ni, mi, nj, mj, nk, mk, nl, ml) = index
        # i > j is implied here
        ni = ni*nMax*nMax*nMax + nj*nMax*nMax + nk*nMax + nl
        mi = mi*nMax*nMax*nMax + mj*nMax*nMax + nMax*mk + nl
        # sort
        if ni < mi: t = ni ; ni = mi ; mi = t
        t = ni*(ni+1)/2 + mi
        assert t < self.shape[0]
        return (<N.float_t*>self.data)[t]

    def __repr__(self, ):
        return "Cache4D(\n      " + (
                    N.ndarray.__repr__(N.asarray(self)))[6:]

    def __str__(self, ):
        return repr(self)

