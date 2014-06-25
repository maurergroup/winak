# thctk.numeric.Combinatorics
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2007 Christoph Scheurer
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
    This Module provides functions for handling combinatorial problems
"""

from copy import copy, deepcopy
from thctk.numeric import * 
from thctk.numeric.SparseMatrix import CSR, CSRd, largeLL
from pysparse import spmatrix

def cartesianProductIterator(*sets):
    wheels = map(iter, sets) # wheels like in an odometer
    digits = [it.next() for it in wheels]
    while True:
        yield digits[:]
        for i in range(len(digits)-1, -1, -1):
            try:
                digits[i] = wheels[i].next()
                break
            except StopIteration:
                wheels[i] = iter(sets[i])
                digits[i] = wheels[i].next()
        else:
            break

def recursiveIndex(length, depth):
    """ 
    Generator for index tuples for i>j>k... with i in range(0,length) and 
    number of indices = 'depth'
    """
    if depth == 0:
        yield ()
    else:
        for i in xrange(length):
            for j in recursiveIndex(length = i, depth = depth-1):
                yield (i,) + j

def reverseIndex(length, depth):
    """ 
    Generator for index tuples for i<j<k... with i in range(0,length) and 
    number of indices = 'depth'
    """
    if depth == 0:
        yield ()
    else:
        for i in xrange(length):
            for j in reverseIndex(length = i, depth = depth-1):
                yield j + (i,)

def xrangeWithout(n, m = None, without = None):
    """
    Generator for the range: 0, 1, ... i-1, i+1,..., n-1
    or the range n, n+1, ..., i-1, i+1, ... m-1
    """
    assert without is not None
    i = without
    if m == None:
        (n, m) = (0, n)
    assert n <= m
    if i < n or i >= m:
        for j in xrange(n, m):
            yield j
        raise StopIteration
    for j in xrange(n, i):
        yield j
    for j in xrange(i+1, m):
        yield j

class Permute:
    """
    Permute a symmetric (sparse) matrix or a vevtor according to permutation 
    vector 'perm'.
    WARNING: This implementation is not efficient, but works for small 
    matrices.
    """
    def __init__(self, perm):
        """
        Initialize using a permutation vector, which is a vector of indices.
        """
        self.perm = perm
        # generate a sequence to be ordered later.
        sortMe = zip(perm, range(len(perm)))
        sortMe.sort()
        self.sortMe = zip(*sortMe)[1]
        # now the permutation can be applied by sorting 'sortMe'

    def __call__(self, A, type = None, threshold = None, columnsOnly = False):
        """
        Apply the permutation on the Matrix A. 'A' can be an 'ndarray' or a 
        sparse matrix of classes 'CSR', 'CSRd' and 'largeLL'.
        Parameters are:
        'type'          'type' will be passed into constructor of CSR, if a CSR 
                        matrix is constructed.
        'threshold'     If a sparse matrix is constructed, all values with 
                        absolute value smaller than threshold will be set to 
                        zero. 
        'columnsOnly'   If set to 'True', only the columns of the matrix A 
                        will be permuted. Otherwise columns and rows will be
                        permuted.
                        If only columns are permuted, A may also be a vector or
                        a non square matrix.
        """
        n = A.shape[0]
        perm = self.perm
        sortMe = self.sortMe
        assert isinstance(A, (CSR, CSRd, largeLL, N.ndarray)), (
            "Matrix A is not instance of class CSR, CSRd, largeLL or N.ndarray")
        if getattr(A, 'ndim', 0) >= 1: 
            if A.ndim == 1: # permute a vector
                return A.take(perm) # permute elements of a vector
            elif A.ndim == 2 and columnsOnly:
                return A.take(perm, axis = 1) # permute only columns
            elif A.ndim == 2 and not columnsOnly:
                assert n == A.shape[0] == A.shape[1]
                assert n == len(perm)
                if threshold is None:
                    B = N.take(A, perm, axis = 0)
                    try:
                        B = N.take(B, perm, axis = 1, out = B)
                    except TypeError:
                        # in place take won't work e.g. for a masked array
                        B = N.take(B, perm, axis = 1)
                    return B
            else: 
                raise TypeError('Array must be 1D or 2D.')

        assert n == A.shape[0] == A.shape[1]
        assert n == len(perm)
        # convert sparse matrix to a python linked list
        l = [[] for i in range(n)]
        try: # assume A is a sparse matrix
            for i, j, x in A:
                l[i].append((j, x))
            nnz = A.nnz
        except: 
            nnz = 0
            print("WARNING: converting matrix to sparse matrix " + 
                    "using 'threshold'.")
            for i in xrange(n):
                if N.abs(A[i,i]) > threshold:
                    l[i].append((i, A[i,i]))
                    nnz += 1
                for j in xrange(i):
                    if N.abs(A[i,j]) > threshold:
                        l[i].append((j, A[i,j]))
                        l[j].append((i, A[j,i]))
                        nnz += 1
        if columnsOnly:
            # don't permute rows
            l = zip(range(len(l)), l)
        else:
            # permute rows
            l = zip(sortMe, l)
            # l.sort()

        for k in xrange(len(l)):
            (i, li) = l[k]
            s = [(sortMe[j], x) for (j,x) in li]
            # s.sort()
            l[k] = (i, s)
        # now columns have been permuted
        # finally construct new sparse matrix
        B = spmatrix.ll_mat(n, n, nnz)
        for (i, li) in l:
            for (j, x) in li:
                B[i,j] = x
        B = B.to_csr()
        # TODO: deepcopy shouldn't be necessary when bug in pysparse is removed
        (mx, mj, mi) = deepcopy(B.matrices())
        if type is not None:
            B = CSR(i = mi, j = mj, x = mx, issym = 1, type = type)
        else:
            B = CSR(i = mi, j = mj, x = mx, issym = 1)
        return B

if __name__ == '__main__':
    pass
