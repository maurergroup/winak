# thctk.numeric.SparseMatrix
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
    This Module provides functions for handling sparse matrices.
"""

from INTERNALS.curvilinear.numeric import *
import copy, pysparse
from types import IntType
from INTERNALS.curvilinear.numeric.spmatrixIterator import spmatrixIterator
from INTERNALS.curvilinear.numeric.csrVSmsr import csrmsr, csrcsc
from INTERNALS.curvilinear.numeric._numeric import amux_CSR, amux_CSR_complex, amux_CSRd
from INTERNALS.curvilinear.numeric._numeric import bosonelements, copyArray, dp_index_dd
from INTERNALS.curvilinear.numeric.rcm import genrcm
from pysparse import spmatrix
try: # this is only built if PARDISO is available
    import thctk.numeric.pardiso
except:
    pass
from scipy.linalg.fblas import dnrm2 as norm2
from scipy.linalg.fblas import ddot
from INTERNALS.curvilinear.numeric import blassm
from INTERNALS.curvilinear.numeric.comm import comrr, comzz, comrz
import itertools
import re

arrayType = type(N.array(()))
LLMatType = type(spmatrix.ll_mat_sym(1,1))
SSSMatType = type(spmatrix.ll_mat_sym(1,1).to_sss())
CSRMatType = type(spmatrix.ll_mat_sym(1,1).to_csr())


class PARDISOError(Exception):

    def __init__(self, value):
        self.value = int(abs(value))
        self.errors = [ None,
            """Input inconsistent""",
            """Not enough memory""",
            """Reordering problem""",
            """Zero pivot, numerical fact. or iterative refinement problem""",
            """Unclassified (internal) error""",
            """Preordering failed (matrix types 11, 13 only)""",
            """Diagonal matrix problem""",
            """32-bit integer overflow problem""",
            ]

    def __str__(self):
        return self.errors[self.value]

class PARDISO:

    def __init__(self, A, mtype, perm = None, np = 1, msglevel = 0,
            Cindex = True):
        self.mtype = mtype
        if Cindex:
            A.i += N.array(1, nxInt32)
            A.j += N.array(1, nxInt32)
        self.A = A
        self.n = len(self.A.i) - 1
        self.nnz = len(self.A.x)
        self.x = N.zeros(self.n, nxFloat)
        self.pt, self.iparm = thctk.numeric.pardiso.pardisoinit(mtype)
        self.iparm[2] = np # number of processors
        self.msglevel = msglevel
        if perm is None:
            self.perm = N.zeros(1)

    def __call__(self, b):
        self.reorder()
        self.factor()
        return self.solve(b)

    def solve(self, b):
        A = self.A
        x, error = thctk.numeric.pardiso.pardiso(self.pt, self.mtype, A.x,
                A.i, A.j, self.perm, self.iparm, b, self.x, phase=33,
                msglvl=self.msglevel)
        if error != 0:
            raise PARDISOError(error)
        return x

    def reorder(self):
        A = self.A
        x, error = thctk.numeric.pardiso.pardiso(self.pt, self.mtype, A.x,
                A.i, A.j, self.perm, self.iparm, self.x, self.x, phase=11,
                msglvl=self.msglevel)
        if error != 0:
            raise PARDISOError(error)

    def factor(self):
        A = self.A
        x, error = thctk.numeric.pardiso.pardiso(self.pt, self.mtype, A.x,
                A.i, A.j, self.perm, self.iparm, self.x, self.x, phase=22,
                msglvl=self.msglevel)
        if error != 0:
            raise PARDISOError(error)

class Preconditioner:
    """
    This is a container class for numerical preconditioners. By default it
    performs the identity mapping. To implement an actual preconditioner one
    needs to implement a mapping function and assign it to the object:

    p = Preconditioner()
    p.mapping = mapping

    where mapping is a function with three arguments: mapping(object, x, y)
    The preconditioner p itself is passed as 'object', and x and y are the
    input and output vectors, respectively. Any additional data that the
    mapping function needs can thus be passed as attributes to p.

    Algorithms using a preconditioner should call these objects either as
    y = p(x, [y]) or p.precon(x, y)
    """

    def __init__(self, mapping = None):
        if mapping is None:
            self.mapping = lambda p, x, y: copyArray(x, y)
        else:
            self.mapping = mapping

    def __call__(self, x, y = None):
        if y is None: y = N.zeros(x.shape, nxTypecode(x))
        self.precon(x, y)
        return y

    def precon(self, x, y):     # pysparse convention
        self.mapping(self, x, y)

class CSR:
    """Matrix A in compressed sparse row format: A.i, A.j, A.x
    n: number of rows
    nnz: length of non-zero entries
    i: integer array of length n+1, pointers to the begining of each row
       (the last entry should be nnz)
    j: integer array of length nnz column indices
    x: double precision array of length nnz, matrix entries row by row
    """

    def __init__(self, n = None, m = None, nnz = None, i = (), j = (), x = (), 
        type = nxFloat, offset = 0, issym=0, filename = None, complexMatVec=False):


        if n is None: # assume data in 'i' ist correct and get 'n' from 'i'.
            n = len(i)-1
        if nnz is None: # nnz should be the last entry in 'i' and 'len(j)'
            nnz = i[-1]
            assert nnz == len(j)
        if len(i) == n+1: self.i = N.asarray(i, nxInt32)
        else: self.i = N.zeros(n+1, nxInt32)
        if len(j) == nnz: self.j = N.asarray(j, nxInt32)
        else: self.j = N.zeros(nnz, nxInt32)
        if len(x) == nnz: self.x = N.asarray(x, type)
        else: self.x = N.zeros(nnz, type)
        self.offset = offset
        self.dtype = type
        self.typecode = type
        self.rows = n
        self.nnz = nnz
        self.issym = issym
        self.n = n
        if m == None: m = n
        self.m = m
        self.shape = (n,m)
        self.setMatVec(complexMatVec)

    def setMatVec(self, complexMatVec=False):
        self.complexMatVec = complexMatVec
        if complexMatVec:
            self.matvec = self.matvecComplex
        else:
            self.matvec = self.matvecReal

    def matrices(self):
        return self.x, self.j, self.i
  
    def __call__(self, x, y = None):
        n = self.n
        if y is None: y = N.zeros(self.n, nxTypecode(x))
        self.matvec(x, y)
        return y

    def __getitem__(self, key):
        i, j = key
        for k in range(self.i[i], self.i[i+1]):
            if self.j[k] == j: return self.x[k]
        return 0

    def __iter__(self):
        i = 0   
        a = self.i[0]
        for b in self.i[1:]:
            for k in range(a, b): 
                yield i, int(self.j[k]), self.x[k]
            a = b
            i += 1
  
    def to_csr(self):
        n = self.n
        nnz = self.nnz
        H = spmatrix.ll_mat(n,n, nnz)
        for i,j,h in self:
            H[i,j] = h
        mat = H.to_csr()
        return mat

    def commutator(self, x, y, commutatorFunction=comrz):
        """ if x and y vectors are complex, commutatorFunction
        should be set to 'comrz' - default value.
        if x and y are real vectors, commutatorFunction should
        be set to 'comrr'
        """
        d, j, i = self.x, self.j, self.i
        if self.offset == 0:
            i += 1
            j += 1
            self.offset = 1
        commutatorFunction(i,j,d,x,y, n=self.n, nnz=self.nnz)

    def to_vec(self, issym=None):
        if issym is None: issym = self.issym
        if issym:
            return self.vectorize()
        else:
            selfT = self.transpose()
            return selfT.vectorize()

    def vectorize(self):
        x, i, j, n, m, nnz = self.x, self.i, self.j, self.n, self.m, self.nnz
        complexMatVec = self.complexMatVec
        newJ = j.copy()
        if self.offset:
            newJ -= 1
        for l in range(1, len(i)):
            b, e = i[l-1], i[l]
            newJ[b:e] += (l-1)*n
        newI = N.array([0,nnz], nxInt32)
        newM = n**2
        newN = 1
        return CSR(n=newN, m=newM, nnz=nnz, i=newI, j=newJ, x=x, complexMatVec=complexMatVec)

    def transpose(self):
        x, i, j, n, m = self.x, self.i, self.j, self.n, self.m
        complexMatVec = self.complexMatVec
        if not self.offset:
            i += 1
            j += 1
        xo, jo, io = csrcsc(n,m,x,j,i)
        i -= 1
        j -= 1
        jo -= 1
        io -= 1
        io = io[:m+1]
        self.offset = 0
        return CSR(n=self.m, m=self.n, nnz=self.nnz, i=io, j=jo, x=xo, complexMatVec=complexMatVec)
 
    def permuted(self, perm = None):
        if perm is None: delattr(self, 'perm')
        else: self.perm = perm

    def cols(self, cols = None):
        if hasattr(self, 'shape'):
            return self.shape[1]
        else:
            if cols is None: cols = max(self.j) + 1
            self.shape = [len(self.i) - 1, cols]
            return cols

    def full(self, cols = None):
        if not hasattr(self, 'shape'): self.cols(cols)
        m = N.zeros(self.shape, nxTypecode(self.x))
        for i in range(self.shape[0]):
            for k in range(self.i[i]-self.offset, self.i[i+1]-self.offset):
                m[i,self.j[k]-self.offset] = self.x[k]
        return m

    def matvecComplex(self, x, y):
        if hasattr(self, 'perm'):
            amux_CSR_complex(self.x, self.j, self.i, x, y, self.offset, self.perm)
        else:
            amux_CSR_complex(self.x, self.j, self.i, x, y, self.offset)

    def matvecReal(self, x, y):
        if hasattr(self, 'perm'):
            amux_CSR(self.x, self.j, self.i, x, y, self.offset, self.perm)
        else:
            amux_CSR(self.x, self.j, self.i, x, y, self.offset)

    def graph(self):
        i = 0
        k = 0
        I = N.zeros(self.i.shape, nxInt)
        J = N.zeros(self.j.shape, nxInt)
        a = self.i[0]
        for b in self.i[1:]:
            for j in self.j[a:b]:
                # the adjacency graph does not contain diagonal elements
                if j != i:
                    J[k] = j
                    k += 1
            i += 1
            I[i] = k
            a = b
        return I, J[:k]
        
    def rowScaling(self):
        self.rowNorms = N.zeros(self.rows, nxFloat)
        i = 0
        while i < self.rows:
            r = self.x[self.i[i]:self.i[i+1]]
            n = N.sqrt(N.sum(r*r))
            self.rowNorms[i] = n
            r /= n
            i += 1

    def scaleRow(self, i, d):
        """
        Scale row 'i' with factor 'd'.
        """
        row = self.x[self.i[i]:self.i[i+1]]
        row *= d

class CSRd:
    """
    Symmetric matrix A in compressed sparse row format with separate diagonal:
    A.i, A.j, A.x, A.d"""

    def __init__(self, n = None, nnz = None, i = (), j = (), x = (), d=(), 
        type = nxFloat, offset = 0, filename = None):
        
        if len(i) == n+1: self.i = N.asarray(i, ncInt32)
        else: self.i = N.zeros(n+1, nxInt32)
        if len(j) == nnz: self.j = N.asarray(j, ncInt32)
        else: self.j = N.zeros(nnz, nxInt32)
        if len(x) == nnz: self.x = N.asarray(x, type)
        else: self.x = N.zeros(nnz, type)
        if len(d) == n: self.d = N.asarray(d, type)
        else: self.d = N.zeros(n, type)
        self.shape = (n,n)
        self.offset = offset
        self.dtype = type
        self.typecode = type
        self.rows = n
        self.nnz = nnz

    def matrices(self):
        return self.d, self.x, self.j, self.i
  
    def __call__(self, x, y = None):
        if y is None: y = N.zeros(self.rows, nxTypecode(self))
        self.matvec(x, y)
        return y

    def __getitem__(self, key):
        i, j = key
        if i == j: return self.d[i]
        for k in range(self.i[i], self.i[i+1]):
            if self.j[k] == j: return self.x[k]
        return 0

    def cols(self):
        return self.shape[1]

    def permuted(self, perm = None):
        if perm is None: delattr(self, 'perm')
        else: self.perm = perm

    def full(self, lower = 1):
        m = N.zeros(self.shape, nxTypecode(self.d))
        for i in range(self.shape[0]):
            m[i,i] = self.d[i]
            for k in range(self.i[i]-self.offset, self.i[i+1]-self.offset):
                j = self.j[k]-self.offset
                m[i,j] = self.x[k]
                if lower: m[j,i] = self.x[k]
        return m

    def matvec(self, x, y):
        if hasattr(self, 'perm'):
            amux_CSRd(self.x, self.d, self.j, self.i, x, y, self.offset, self.perm)
        else:
            amux_CSRd(self.x, self.d, self.j, self.i, x, y, self.offset)

    matvec_transp = matvec

    def toCSR(self):
        m = self.rows
        B = CSR(m, None, m + 2*len(self.j), type = nxTypecode(self.x))
        B.shape = copy.copy(self.shape)
        ind = N.zeros(m+1)
        for j in self.j:
            ind[j] += 1
        for i in range(m):
            a = self.i[i]
            b = self.i[i+1]
            n = b - a
            c = B.i[i] + ind[i]
            B.i[i+1] = c + n + 1
            B.j[c] = i
            B.j[c+1:B.i[i+1]] = self.j[a:b]
            B.x[c] = self.d[i]
            B.x[c+1:B.i[i+1]] = self.x[a:b]
        ind[:] = B.i
        for i in range(m):
            for k in range(self.i[i], self.i[i+1]):
                j = self.j[k]
                B.j[ind[j]] = i
                B.x[ind[j]] = self.x[k]
                ind[j] += 1
        return B

    def graph(self):
        m = self.rows
        I = N.zeros(self.i.shape, nxInt)
        J = N.zeros(2*len(self.j), nxInt)
        ind = N.zeros(self.i.shape, nxInt)
        for j in self.j:
            ind[j] += 1
        for i in range(m):
            a = self.i[i]
            b = self.i[i+1]
            c = I[i] + ind[i]
            I[i+1] = c + b - a
            J[c:I[i+1]] = self.j[a:b]
        ind[:] = I
        for i in range(m):
            for k in range(self.i[i], self.i[i+1]):
                j = self.j[k]
                J[ind[j]] = i
                ind[j] += 1
        return I, J

def CSR2pbm(A, offset = 0, pbm = 'CSR.pbm', s = None, comment = ''):
    """
    CSR2pbm(A, offset = 0, pbm = 'CSR.pbm', s = None)

    This function takes a sparse matrix in CSR format and produces a binary
    PBM file of its sparsity pattern. If s is given, it is supposed to behave
    like a file object, i.e. s.write(), s.flush(), and s.close() have to be
    implemented. This can be used to directly compress the bitmap, e.g. to a
    G3- or G4-encoded format (see pnmtotiff, pbmtog3, and pbmtopsg3).

    Matrices in CSRd format can also be processed but only the upper triangle
    will be shown. Transposing of the bitmap can be achieved using 
        pnmflip -transpose A.pbm > At.pbm
    and composing the full bitmap is done by (white = 1, black = 0)
        pnmpaste -and A.pbm 0 0 At.pbm > Afull.pbm

    For Fortran index convention use offset = 1. The comment is included
    verbatim in the PBM file and should therefore adhere to the PBM file
    conventions.
    """
    if s is None:
        s = open(pbm, 'w')
    m, n = A.shape
    s.write('P4\n# sparse matrix structure plot\n')
    if comment: s.write(comment)
    s.write('\n# nnz = %d\n%d %d\n' % (len(A.j), n, m))
    s.flush()
    empty = '\x00' * ((n-1)/8 + 1)
    i = 0
    b = A.i[i] - offset
    while i < m:
        a = b
        i += 1
        b = A.i[i] - offset
        if a == b:
            s.write(empty)
        else:
            l = list(A.j[a:b]-offset)
            l.sort()
            c = 0
            bb = 8
            for j in l:
                q = (j - bb)%8
                p = (j - bb)/8
                if p > 0:
                    s.write(chr(c))
                    c = 128 >> q
                    bb = (j/8 + 1)*8
                    s.write(empty[:p])
                elif p < 0:
                    c |= 128 >> q
                else:
                    s.write(chr(c))
                    c = 128 >> q
                    bb += 8
            s.write(chr(c))
            if bb < n:
                s.write(empty[:(n-bb-1)/8+1])
        s.flush()
    s.close()

def rcm(A):
    """ p = rcm(A)
        
        generate the reverse Cuthill-McKee ordering of A
    """
    i, j = A.graph()
    if A.offset == 0: # genrcm expects fortran indexing
        i += 1
        j += 1
    p = genrcm(i, j)
    p -= 1
    return p

class Sparse:

# The following three methods should be implemented by derived classes:
#
#   __init__        initialization, always call Sparse.__init__(self)
#                   to initialize the transpose, hermitian, and complex
#                   conjugate
#   __getitem__     retrieve matrix elements A[i,j] also for transpose, etc.
#   __call__        implement matrix-vector multiplication as A(v)
#   __len__         return the number of nonzero elements
#   getIndex        return the nonzero elements as an Index sparse matrix
#

    def __init__(self, typecode = nxFloat):
        self.shape = []
        self.transpose = 0
        self.conjugate = 0
        self.MT = None
        self.MC = None
        self.MA = None
        self.array = None
        self.dtype = typecode
        self.typecode = typecode
        self.SparseIndex = None

    def __getitem__(self, key): return 0

    def __call__(self, v): return v

    def __len__(self): return 0

    def getIndex(self):
        self.SparseIndex = Index([], [], [], [])
        
    def nonzero(self):
        if self.SparseIndex == None: self.getIndex()
        return self.SparseIndex

    def matrix(self):
        # Make this multidimensional and check whether representations for the
        # transpose, etc. already exist. If so, use them.
        if self.array is None:
            self.array = N.zeros(self.shape, self.dtype)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.array[i,j] = self[i,j]
        return self.array

    def cp(self):
        return copy.deepcopy(self)

    def Transpose(self):
        if self.MT is None:
            b = copy.copy(self)
            b.transpose = not b.transpose
            b.shape.reverse()
            b.MT = self
            b.MC = self.MA
            b.MA = self.MC
            self.MT = b
        return self.MT

    T = Transpose

    def ComplexConjugate(self):
        if self.MC is None:
            b = copy.copy(self)
            b.conjugate = not b.conjugate
            b.MC = self
            b.MT = self.MA
            b.MA = self.MT
            self.MC = b
        return self.MC

    CC = ComplexConjugate

    def Adjoint(self):
        if self.MA is None:
            b = copy.copy(self)
            b.transpose = not b.transpose
            b.shape.reverse()
            b.conjugate = not b.conjugate
            b.MA = self
            b.MT = self.MC
            b.MC = self.MT
            self.MA = b
        return self.MA
        
    A = Adjoint
    Dagger = Adjoint

def FrobeniusNorm(matrix):
  el = 0.
  for i,j,h in matrix:
    el += h**2
  return N.sqrt(el)

def AbsNorm(matrix):
  el = 0.
  for i,j,h in matrix:
    el += abs(h)
  return el

class Index(Sparse):

    def __init__(self, shape, i, j, a):
        a = N.array(a)
        Sparse.__init__(self, nxTypecode(a))
        self.iindex = N.array(i, nxInt)
        self.jindex = N.array(j, nxInt)
        self.data = a
        self.shape = shape
        self.SparseIndex = self

    def __getitem__(self, key):
        try:
            i = key[0]; j = key[1]
            return 0
        except:
            return self.iindex[key], self.jindex[key], self.data[key]

class Diagonal(Sparse):

    def __init__(self, n, *diagonals):
        Sparse.__init__(self)
        self.shape = [n,n]
        self.index = []
        self.data = []
        e = N.ones(1)
        for d in diagonals:
            i = d[0]
            a = N.array(d[1])
            if len(a) != n - abs(i): raise(IndexError)
            e = e * a[0]
            self.index.append(i)
            self.data.append(a)
        self.dtype = nxTypecode(e)

    def __getitem__(self, key):
        if self.transpose: j = key[0]; i = key[1]
        else: i = key[0]; j = key[1]
        d = j-i
        for k in range(len(self.index)):
            if self.index[k] == d: return self.data[k][min(i,j)]
        return 0

    def __len__(self):
        n = 0
        for k in self.index: n += self.shape[0] - abs(k)
        return n

    def getIndex(self):
        nn = len(self)
        dd = N.zeros(nn, nxTypecode(self))
        di = N.zeros(nn, nxInt)
        dj = N.zeros(nn, nxInt)
        m = 0
        for k in range(len(self.index)):
            j = self.index[k]
            d = self.data[k]
            n = self.shape[0] - abs(j)
            if j < 0:
                i = abs(j)
                j = 0
            else: i = 0
            for l in range(n):
                di[m] = i
                dj[m] = j
                dd[m] = d[l]
                i += 1; j += 1; m += 1
        self.SparseIndex = Index(self.shape, di, dj, dd)

class DiagonalSymmetric(Diagonal):

    def __getitem__(self, key):
        i = key[0]
        j = key[1]
        d = abs(j-i)
        for k in range(len(self.index)):
            if abs(self.index[k]) == d: return self.data[k][min(i,j)]
        return 0

    def __len__(self):
        n = 0
        for k in self.index: 
            if k == 0: n += self.shape[0] - abs(k)
            else: n += 2*(self.shape[0] - abs(k))
        return n

    def getIndex(self):
        nn = len(self)
        dd = N.zeros(nn, nxTypecode(self))
        di = N.zeros(nn, nxInt)
        dj = N.zeros(nn, nxInt)
        m = 0
        for k in range(len(self.index)):
            j = self.index[k]
            d = self.data[k]
            n = self.shape[0] - abs(j)
            if j < 0:
                i = abs(j)
                j = 0
            else: i = 0
            for l in range(n):
                if i != j:
                    di[m] = i
                    dj[m] = j
                    dd[m] = d[l]
                    m += 1
                di[m] = j
                dj[m] = i
                dd[m] = d[l]
                i += 1; j += 1; m += 1
        self.SparseIndex = Index(self.shape, di, dj, dd)

class Boson(Diagonal):

    def __init__(self, n, *pm):
        if type(n) != IntType or int(n) < 1: raise(TypeError)
        if len(pm) == 0:
            p = 0
            m = 0
        elif len(pm) == 1:
            k = int(pm[0])
            p = max(0,k)
            m = abs(min(0,k))
        elif len(pm) == 2:
            p = abs(int(pm[0]))
            m = abs(int(pm[1]))
        i = m - p
        if p == 0 and m == 0:
            d = N.ones(n)
        else:
#           d = N.zeros(n, nxFloat)
#           a = min(p,m)
#           b = max(p,m)
#           for j in range(a, n):
#               d[j] = 1
#               for k in range(j-a+1,j+1):     d[j] *= k
#               for k in range(j-a+1,j+b-a+1): d[j] *= k
#           d = N.sqrt(d[:n-abs(i)])
            d = bosonelements(n, p, m)
        Diagonal.__init__(self, n, (i, d))
        self.p = p
        self.m = m
        self.n = n

    def __len__(self):
        return self.n - abs(self.p - self.m) - min(self.p, self.m)

    def getIndex(self, typecode = None):
        n = len(self)
        p = self.p
        m = self.m
        if typecode is None: typecode = nxTypecode(self)
        a = self.data[0][self.n - abs(p-m) - n:].astype(typecode)
        i = N.arange(n) + (max(0, p-m) + min(p,m))
        j = N.arange(n) + (max(0, m-p) + min(p,m))
        self.SparseIndex = Index(self.shape, i, j, a)

class TensorProduct(Sparse):

    def __init__(self, r, l, a, typecode = None):
        if typecode is None: 
            typecode = nxTypecode(a)
        Sparse.__init__(self, typecode)
        self.shape = []
        n = r * l
        for i in a.shape: self.shape.append(n*i)
        self.data = a
        self.r = r
        self.l = l

    def __len__(self):
        n = len(self.data)
        return self.r * self.l * n

    def __getitem__(self, key):
        if self.transpose:
            i = key[1]
            j = key[0]
        else:
            i = key[0]
            j = key[1]
        l = self.l
        r = self.r
        m = self.data.shape[0]
        n = self.data.shape[1]
        if i/(l*m) != j/(l*n): return 0
        i = i % (l*m)
        j = j % (l*n)
        if i%l != j%l: return 0
        return self.data[i/l, j/l]

    def __call__(self, x, y = None, c = 1.):
        if y is None: y = N.zeros(x.shape, nxTypecode(x))
        a = self.data.nonzero()
        dp_index_dd(a.data, a.shape[0], a.shape[1], a.iindex, a.jindex,
            self.transpose, self.l, self.r, c, x, y)
        return y

def AmuB(A, B, Cnnz=None, job=1, offset=0):
    """
    performs the matrix by matrix product C = A*B
    in csr format
    if job is 0 only structure of C will be calculated (cj and ci arrays
    both matrices should have the sane offset
    """
    ax, aj, ai = A.matrices()
    bx, bj, bi = B.matrices()
    if not offset:
      aj += 1
      ai += 1
      bj += 1
      bi += 1
    an, am = A.shape
    bn, bm = B.shape
    if am != bn: raise typeError, 'ncol(A) must be equal nrow(B)!'
    if Cnnz is None: Cnnz = an*bm+1
    cx, cj, ci, ierr = blassm.amub(bm, ax, aj, ai, bx, bj, bi, Cnnz, job=job)
    aj -= 1
    ai -= 1
    bj -= 1
    bi -= 1
    cj -= 1
    ci -= 1
    cx = cx[:ci[-1]]
    cj = cj[:ci[-1]]
    return CSR(n=(len(ci)-1), nnz=len(cx), x=cx, j=cj, i=ci, offset=0)

def AplsB(A, B, s=1, Cnnz=None, offset=0):
    """
    performs the matrix by matrix addition C = A + s * B
    
    in csr format
    both matrices should have the sane offset
    """
    ax, aj, ai = A.matrices()
    bx, bj, bi = B.matrices()
    if not offset:
      aj += 1
      ai += 1
      bj += 1
      bi += 1
    an, am = A.shape
    if Cnnz is None: Cnnz = an*am+1
    cx, cj, ci, ierr = blassm.aplsb(am, ax, aj, ai, s, bx, bj, bi, Cnnz)
    aj -= 1
    ai -= 1
    bj -= 1
    bi -= 1
    cj -= 1
    ci -= 1
    cx = cx[:ci[-1]]
    cj = cj[:ci[-1]]
    return CSR(n=(len(ci)-1), m=None, nnz=len(cx), x=cx, j=cj, i=ci, offset=0)
  
def AmS(A, s):
    for i, j, h in A:
      h *= s
    return A 
  
def Bp(L):
    """ b = Bp(L)
        
        returns a matrix for the operator B^dagger for an L-level system
    """
    b = N.zeros((L,L),nxFloat)
    for i in range(1,L):
        b[i,i-1] = N.sqrt(i)
    return b

def Bm(L):
    """ b = Bm(L)
        
        returns a matrix for the operator B for an L-level system
    """
    b = N.zeros((L,L),nxFloat)
    for i in range(1,L):
        b[i-1,i] = N.sqrt(i)
    return b

def pythag(x, y):
    return N.sqrt(x*x + y*y)

def lsqr(A, b, niter = 0, x = None, damp = 0, condlim = 0, atol = 0, btol = 0,
         Kr = None, Kl = None, debug = 0):

    """
    x, rho, eta, (info, i, msg, anorm, arnorm, acond) =
        lsqr(A, b, niter = 0, x = None, damp = 0, condlim = 0, atol = 0,
             btol = 0, K = None, debug = 0)

    compute the regularized iterative solution of Ax = b
    rho and eta contain the norms of the solution and residual

    Kr (Kl) can be a right (left) pre-conditioner K which has to
    implement the call K.precon(x, y) for y = Kx

    Reference: C. C. Paige & M. A. Saunders, "LSQR: an algorithm for 
    sparse linear equations and sparse least squares", ACM Trans. 
    Math. Software 8 (1982), 43-71.

    see also the Matlab implementation at:
        http://www.stanford.edu/group/SOL/software/lsqr/matlab/lsqr.m
    """

    Ax  = A.matvec
    Atx = A.matvec_transp
    m, n = A.shape
    if niter <= 0:
        niter = min(m, n)
    info = 0

    msg = [
        'The iteration limit has been reached',
        'Ax - b is small enough, given atol, btol',
        'The least-squares solution is good enough, given atol',
        'The estimate of cond(Abar) has exceeded condlim',
        'Ax - b is small enough for this machine',
        'The least-squares solution is good enough for this machine',
        'Cond(Abar) seems to be too large for this machine',
    ]

    # Prepare for LSQR iteration
    if nxTypecode(b) != nxFloat or len(b) != m:
        raise TypeError("Vector b has wrong type or length")
    bnorm = norm2(b)
    if bnorm == 0:
        raise ValueError("Vector b must be nonzero")

    # solution vector
    if x is None:
        x = N.zeros(n, nxFloat)
    else:
        if nxTypecode(x) != nxFloat or len(x) != n:
            raise TypeError("Vector x has wrong type or length")
        x[:] = 0

    # auxiliary vectors
    p = N.zeros(m, nxFloat)
    r = N.zeros(n, nxFloat)

    # the Preconditioner needs one extra storage vector
    if Kr is not None:
        rv = N.zeros(x.shape, nxTypecode(x))

    if Kl is not None:
        lv = N.zeros(b.shape, nxTypecode(b))

    # Initialize the bi-diagonalization
    if Kl is not None:      # use the left pre-conditioner
        u = N.zeros(b.shape, nxTypecode(b))
        Kl.precon(b, u)
        bnorm = norm2(u)
        u /= bnorm
    else:
        u = b/bnorm
    beta = bnorm
    Atx(u, r)
    if Kr is not None:      # use the right pre-conditioner
        Kr.precon(r, rv)
        tmpv = r
        r = rv
        rv = tmpv
    alpha = norm2(r)
    v = r/alpha
    w = copy.copy(v)
    rho_bar = alpha
    phi_bar = beta

    # initialize variables for the computation of norms
    eta = []
    rho = []
    res2 = 0
    c2 = -1
    s2 = 0
    z = 0
    xnorm = 0
    bbnorm = 0
    ddnorm = 0
    xxnorm = 0
    rnorm = beta
    arnorm = alpha*beta
    dampsq = damp*damp
    ctol = 0
    if condlim > 0:
        ctol = 1/condlim

    if debug:
        s = "%3s%10s%10s%10s%10s%10s%10s%10s\n" % ('nit', 'alpha', 'beta',
            'rrho', 'phi', 'theta', 'rho_bar', 'phi_bar')
        s += '-' * (len(s)-1)
        print s

    i = 1
    while i < niter and info == 0:
        i += 1
        alpha_old = alpha
        beta_old = beta

        # Continue the bi-diagonalization
        if Kr is not None:      # use the right pre-conditioner
            Kr.precon(v, rv)
            tmpv = v
            v = rv
            rv = tmpv
        Ax(v, p)
        u *= alpha
        p -= u                  # p = A v - alpha u
        beta = norm2(p)
        u = N.divide(p, beta, u)

        if Kl is not None:      # use the left pre-conditioner
            Kl.precon(u, lv)
            tmpv = u
            u = lv
            lv = tmpv
        Atx(u, r)
        v *= beta
        r -= v                  # r = A^T u - beta v
        alpha = norm2(r)
        v = N.divide(r, alpha, v)

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rho_bar) of the lower-bidiagonal matrix.

        rho_bar1 = pythag(rho_bar, damp)
        c1       = rho_bar / rho_bar1
        s1       = damp / rho_bar1
        psi      = s1 * phi_bar
        phi_bar  = c1 * phi_bar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.

        rrho     = pythag(rho_bar1, beta)
        c1       = rho_bar1 / rrho
        s1       = beta / rrho
        theta    =  s1 * alpha
        rho_bar  = -c1 * alpha
        phi      =  c1 * phi_bar
        phi_bar  =  s1 * phi_bar

        if debug:
            print "%3d%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f" % \
                (i, alpha, beta, rrho, phi, theta, rho_bar, phi_bar)

        # Update x and w
        w /= rrho
        r = N.multiply(w, phi, r)
        x += r                      # x = x + (phi/rrho) * w
        ddnorm += ddot(w, w)
        w *= -theta
        w += v                      # w = v - (theta/rrho) * w

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate  norm(x).

        delta = s2*rrho
        gamma_bar = -c2*rrho
        rhs = phi - delta*z
        z_bar = rhs/gamma_bar
        xnorm = N.sqrt(xxnorm + z_bar*z_bar)
        gamma = pythag(gamma_bar, theta)
        c2 = gamma_bar/gamma
        s2 = theta/gamma
        z = rhs/gamma
        xxnorm += z*z

        eta.append(xnorm)

        # Compute norms for convergence criteria (see TOMS 583)
        bbnorm += alpha_old*alpha_old + beta*beta
        anorm = N.sqrt(bbnorm)
        acond = anorm * N.sqrt(ddnorm)
        arnorm = alpha * abs(s1*phi)
        res1 = phi_bar * phi_bar
        res2 += psi * psi
        rnorm = N.sqrt(res1 + res2)

        rho.append(rnorm)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.

        r1sq   = rnorm*rnorm - dampsq*xxnorm
        r1norm = N.sqrt(abs(r1sq))
        if r1sq < 0: r1norm = -r1norm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = rnorm/bnorm
        test2 = arnorm/(anorm*rnorm)
        test3 = 1/acond
        t1    = test1/(1 + anorm*xnorm/bnorm)
        rtol  = btol + atol*anorm*xnorm/bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  condlim = 1/eps.

        if 1 + test3  <= 1: info = 6
        if 1 + test2  <= 1: info = 5
        if 1 + t1     <= 1: info = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol: info = 3
        if test2 <= atol: info = 2
        if test1 <= rtol: info = 1

    if Kr is not None:
        Kr.precon(x, rv)
        x[:] = rv

    return x, rho, eta, (info, i, msg[info], anorm, arnorm, acond)

def cgls(A, b, niter = 0, x = None, condlim = 0, atol = 0, btol = 0,
         Kl = None, Ki = None, Kr = None, debug = 0):
    """
    preconditioners:
        Kl      ... left preconditioner:  Kl At (A x - b) = 0
        Kr      ... right preconditioner: At (A Kr x - b) = 0
        Kl & Kr ... split preconditioner: Kl At (A Kr x - b) = 0
        Ki      ... inner preconditioner: At Ki (A x - b) = 0
    """

    swap = lambda x,y: (y,x)
    Ax  = A.matvec
    Atx = A.matvec_transp
    m, n = A.shape
    if niter <= 0:
        niter = min(m, n)
    info = 0

    if debug:
        fmt = "%4d " + "%16.6e" * 4

    # Prepare for CGLS iteration
    if nxTypecode(b) != nxFloat or len(b) != m:
        raise TypeError("Vector b has wrong type or length")
    bnorm = norm2(b)
    if bnorm == 0:
        raise ValueError("Vector b must be nonzero")

    # solution vector, we use x0 = 0
    if x is None:
        x = N.zeros(n, nxFloat)
    else:
        if nxTypecode(x) != nxFloat or len(x) != n:
            raise TypeError("Vector x has wrong type or length")
        x[:] = 0

    # auxiliary vectors
    s = N.zeros(n, nxFloat)
    q = N.zeros(m, nxFloat)

    # the Preconditioners need one extra storage vector
    if Kl is not None or Kr is not None:
        vn = N.zeros(x.shape, nxTypecode(x))

    if Ki is not None:
        vm = N.zeros(b.shape, nxTypecode(b))

    # initialize CG iteration
    if Ki is not None:
        r = N.zeros(b.shape, nxFloat)
        Ki.precon(b, r)
    else:
        r = copy.copy(b) # since: x0 = 0  and thus also: Kr x0 = 0
    bnorm = norm2(r)
    Atx(r, s)
    if Kl is not None:
        p = N.zeros(s.shape, nxFloat)
        Kl.precon(s, p)
    else:
        p = copy.copy(s)
    gamma = ddot(s, p)

    # continue CG iteration
    i = 1
    while i < niter and info == 0:
        i += 1
        if Kr is not None:
            Kr.precon(p, vn)
            p, vn = swap(p, vn)
        Ax(p, q)
        if Ki is not None:
            Ki.precon(q, vm)
            q, vm = swap(q, vm)
            alpha = gamma/ddot(q, vm)
        else:
            alpha = gamma/ddot(q, q)
        x += N.multiply(p, alpha, s)    # temporary use of s
        r -= N.multiply(q, alpha, q)
        Atx(r, s)
        if Kl is not None:
            Kl.precon(s, vn)
            s, vn = swap(s, vn)
            gamma_new = ddot(s, vn)
        else:
            gamma_new = ddot(s, s)
        beta = gamma_new/gamma
        if debug:
            print fmt % (i, alpha, beta, gamma, gamma_new)
        gamma = gamma_new
        p *= beta
        p += s

        if norm2(r)/bnorm < 1e-8:
            info = 1
        if gamma < 1e-8:
            info = 3

    return x, r, (info, i)

def spmatrix_fromfile(file, shape = None):
    magic = 'SpMatrix'
    file.seek(0)
    s = file.read(len(magic))
    version, mattype = N.fromstring(file.read(2), nxInt8)
    i, j, k = N.fromstring(file.read(12), nxInt32)
    t = mattype & 0x7f
    if t == 2:                  # CSR
        m, n, nnz = i, j, k
        fshape = (m, n)
    elif t == 3:                # SSS
        n, nnz = i, j
        fshape = (n, n)
    else:
        raise IOError("wrong filetype for spmatrix")
    if shape is not None and fshape != shape:
        raise IndexError("wrong shape in spmatrix file")
    if t == 2:                # CSR
        A = pysparse.spmatrix.csr_mat(m, n, nnz)
    elif t == 3:              # SSS
        A = pysparse.spmatrix.sss_mat(n, nnz)
    else:
        A = None
    A.fromfile(file)
    return A


if __name__ == '__main__':
    a = DiagonalSymmetric(5, (0,(1,2,3,4,5)), (-1,(1,2,3,4)))
    b = N.zeros((5,5),nxFloat)
    for i in range(5):
        for j in range(5): b[i,j] = a[i,j]
    p=Bp(5)
    m=Bm(5)
    c = N.array([[1,2],[3,4],[5,6]])
    d = TensorProduct(3,2,c)

    e = TensorProduct(2,2,Boson(3,1,1))
    id = N.zeros(e.shape, e.typecode())
    for i in range(e.shape[0]):
        id[i,i] = 1
        id[i] = e(id[i])
