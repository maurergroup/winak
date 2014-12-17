#!/usr/bin/env python
""" SuiteSparseQR Python wrapper """

import os.path
import ctypes
from ctypes import c_double, c_size_t, byref, pointer, POINTER
import numpy as np
from numpy.ctypeslib import ndpointer

# Assume spqr_wrapper.so (or a link to it) is in the same directory as this file
spqrlib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + os.path.sep + "spqr.so")

# Function prototypes for spqr_wrapper.so
# void qr_solve(double const *A_data, double const *A_row, double const *A_col, size_t A_nnz, size_t A_m, size_t A_n, double const *b_data, double *x_data)
spqrlib.qr_solve.argtypes = [
        ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # A_data
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # A_row
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # A_col
        c_size_t,  # A_nnz
        c_size_t,  # A_m
        c_size_t,  # A_n
        ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # b_data
        ndpointer(dtype=np.float64, ndim=1, flags=('C_CONTIGUOUS', 'WRITEABLE')), # x_data
        ]
spqrlib.qr_solve.restype = None

def qr_solve(A_data, A_row, A_col, A_nnz, A_m, A_n, b_data):
    """ Python wrapper to qr_solve """
    if len(A_data) != len(A_row) != len(A_col) != A_nnz:
        raise TypeError("A_data, A_row, A_col, A_nnz must agree")
    if len(b_data) != A_m:
        raise TypeError("b_data must be A_m long")

    x_data = np.empty(A_n, dtype=np.float64)
    spqrlib.qr_solve(
            np.require(A_data, np.float64, 'C'),
            np.require(A_row, np.int64, 'C'),
            np.require(A_col, np.int64, 'C'),
            A_nnz, A_m, A_n,
            np.require(b_data, np.float64, 'C'),
            np.require(x_data, np.float64, 'C')
            )
    return x_data

def main():
    print("Testing qr_solve")
    A_data = np.array([2, 9, 25], dtype=np.float64)
    A_row = np.array([0, 1, 2])
    A_col = np.array([0, 1, 2])
    b_data = np.array([1, 1, 1], dtype=np.float64)

    x_data = qr_solve(A_data, A_row, A_col, len(A_data), 3, 3, b_data)
    print(x_data)

if __name__ == "__main__":
    main()
