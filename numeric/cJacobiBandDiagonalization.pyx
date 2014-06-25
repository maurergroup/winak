import numpy as N
cimport numpy as N
cimport cython

N.import_array()

# threshold to determine bandwidth of rotation matrix
DEF ZERO = 1.0e-16
# DEF ZERO = 0.0

# block sizes for band matrix multiplication
# DEF ELEMENTS_PER_BLOCK = 64000
# DEF ELEMENTS_PER_BLOCK = 32000
DEF ELEMENTS_PER_BLOCK = 16000

if sizeof(double) != sizeof(N.float_t):
    raise TypeError(
    "Size of 'double' (%i byte) is not equal to size of 'N.float_t' (%i byte)! "
                                %(sizeof(double), sizeof(N.float_t)) + 
    "The cblas routines and the ndarray types won't fit."
    ) 

cdef extern from "numpy/arrayobject.h":
    void* PyArray_GETPTR2(N.ndarray obj, int i, int j)
    void* PyArray_GETPTR3(N.ndarray obj, int i, int j, int k)

cdef extern from "stdlib.h":
    void qsort(void *__base, int __nmemb, int __size, void *compare)

cdef extern from "math.h":
    double cos(double x)
    double sin(double x)
    double atan2(double x, double y)
    double sqrt(double x)
    double fabs(double x)
    int logb(double x)
    int floor(double x)

cdef extern from "cblas.h":
    enum CBLAS_ORDER:     
        CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: 
        CblasNoTrans, CblasTrans, CblasConjTrans

    void cblas_dswap(int n, double *x, int incx, double *y, int incy)
    void cblas_dcopy(int n, double *x, int incx, double *y, int incy)
    int cblas_idamax(int n, double *x, int incx)
    void cblas_dscal(int n, double alpha, double *x, int incx)
    double cblas_dnrm2(int n, double *x, int incx)
    double cblas_ddot(int n, double *x, int incx, double *y, int incy)
    void cblas_daxpy(int n, double alpha, double *x, int incx, 
                        double *y, int incy)
    void cblas_dgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, 
                        int m, int n, double alpha, 
                        double *A, int lda, double *x, int incx, double beta,
                        double *y, int incy)
    void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB, int m, int n, int k,
                        double alpha, double *A, int lda, double *B, int ldb,
                        double beta, double *C, int ldc)

cdef struct SortType_int_int:
    N.int_t a
    N.int_t i

cdef struct SortType_float_int:
    N.float_t a
    N.int_t i

StorageInfoDict = {
    0 : 'NOT_AVAILABLE', 1 : 'LOWER_TRIANGLE', 2 : 'UPPER_TRIANGLE'
    }
cdef enum StorageInfo:
    NOT_AVAILABLE = 0, LOWER_TRIANGLE = 1, UPPER_TRIANGLE = 2

cdef enum FinallyDo:
    FILL_DIAGONAL, FILL, TRAFO

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int cGetBandwidth(N.ndarray Min):
    """
    Calculated bandwidth of a matrix M.
    """
    cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] M = Min
    cdef int bw = 0
    cdef int n = M.shape[0]
    assert M.shape[0] == M.shape[1]
    assert M.ndim == 2

    cdef int i, j
    cdef int l
    for j in range(n):
        l = j - bw
        if l > 0:
            for i in range(l):
                if fabs(M[i,j]) >= ZERO:
                    if j - i > bw: bw = j - i
                    break
        l = j + bw
        if l < n - 1:
            # for whatever reason this loop is not translated properly
            # for i in range(n-1, l, -1): 
            for i from n-1 >= i > l:
                if fabs(M[i,j]) >= ZERO:
                    if i - j > bw: bw = i - j
                    break
    return bw

def getBandwidth(N.ndarray M):
    """
    Calculated bandwidth of a matrix M.
    This is a wrapper for Python, that calls the corresponding Cython function.
    """
    return cGetBandwidth(M)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void cRayleighQuotients(N.ndarray A, N.ndarray R, int bwR, int normR,
                                                            N.ndarray out):
    """
    Compute Rayleigh Quotient values of A using R, i.e. r = R.T*A*R/norm(R)
    """
    cdef int n = out.shape[0]
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] r = out
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] v
    cdef unsigned int i, m, k, nPlusOne
    cdef double *Ri, *Ai, *A0, *v0
    A0 = <double*>A.data

    if True: # WORKAROUND
    # if bwR <= 0 or bwR >= n:
        v = N.empty(n, dtype = N.float)
        v0 = <double*>v.data
        Ri = <double*>R.data
        for i in range(n):
            # v = A*R[:,i]
            cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                        n, n, 1, A0, n, Ri, n, 0., v0, 1)
            # x = R.T[i,:]*v = R[:,i]*v
            r[i] = cblas_ddot(n, Ri, n, v0, 1)
            Ri += 1
        if normR == 1:
            Ri = <double*>R.data
            for i in range(n): 
                r[i] /= cblas_ddot(n, Ri, n, Ri, n)
                Ri += 1
        # assert N.allclose(r, N.diag(N.dot(N.dot(R.T, A), R))/N.diag(N.dot(R.T, R)), 1e-15, 1e-15)
    # else: # TO BE IMPLEMENTED PROPERLY
    #     m = 2*bwR + 1
    #     nPlusOne = n + 1
    #     v = N.empty(n, dtype = N.float)
    #     v0 = <double*>v.data
    #     Ai = <double*>A.data
    #     Ri = <double*>R.data
    #     for i in range(m):
    #         # v = A*R[:,i]
    #         cblas_dgemv(CblasRowMajor, CblasNoTrans, 
    #                     n, i, 1, A0, n, Ri, n, 0., v0, 1)
    #         # x = R.T[i,:]*v = R[:,i]*v
    #         r[i] = cblas_ddot(i, Ri, n, v0, 1)
    #         Ri += 1
    #     Ri += n
    #     for i in range(m, n - m):
    #         # v = A*R[:,i]
    #         cblas_dgemv(CblasRowMajor, CblasNoTrans, 
    #                     n, m, 1, Ai, n, Ri, n, 0., v0, 1)
    #         # x = R.T[i,:]*v = R[:,i]*v
    #         r[i] = cblas_ddot(m, Ri, n, v0, 1)
    #         Ai += 1 ; Ri += nPlusOne
    #     k = m
    #     for i in range(n - m, n):
    #         # v = A*R[:,i]
    #         cblas_dgemv(CblasRowMajor, CblasNoTrans, 
    #                     n, k, 1, Ai, n, Ri, n, 0., v0, 1)
    #         # x = R.T[i,:]*v = R[:,i]*v
    #         r[i] = cblas_ddot(k, Ri, n, v0, 1)
    #         Ai += 1 ; Ri += nPlusOne
    #         k -= 1
    #     if normR == 1:
    #         Ri = <double*>R.data
    #         for i in range(m):
    #             r[i] /= cblas_ddot(i, Ri, n, Ri, n)
    #             Ri += 1
    #         Ri += n
    #         for i in range(m, n - m):
    #             r[i] /= cblas_ddot(m, Ri, n, Ri, n)
    #             Ri += nPlusOne
    #         k = m
    #         for i in range(n - m, n):
    #             r[i] /= cblas_ddot(m, Ri, n, Ri, n)
    #             Ri += nPlusOne
    #             k -= 1
    #     assert N.allclose(r, N.diag(N.dot(N.dot(R.T, A), R))/N.diag(N.dot(R.T, R)), 1e-15, 1e-15)

def rayleighQuotients(N.ndarray A, N.ndarray R, int bwR = 0, normR = True, 
                N.ndarray out = None):
    """
    Compute Rayleigh quotients of A using R, i.e. r = R.T*A*R/norm(R)
    This is a wrapper for Pyhton, that calls the corresponding Cython function.
    """
    cdef int n = A.shape[0]
    assert A.ndim == 2
    assert R.ndim == 2
    assert A.shape[1] == n
    assert R.shape[0] == n
    assert R.shape[1] == n
    if out is None:
        out = N.zeros(n, dtype = N.float)
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] r = out
    assert r.shape[0] == n
    if normR: cRayleighQuotients(A, R, bwR, 1, r)
    else: cRayleighQuotients(A, R, bwR, 0, r)
    return r

cdef inline void cSetSignsOfEigenvectors(N.ndarray R, int bwR):
    """
    Set the signs of the eigenvectors, the largest absolute elment of 
    the vector should be positive.
    """
    cdef unsigned int n = R.shape[0]
    cdef int i, li, ri, bw, # argMax
    cdef double *Ri
    for i in range(n):
        li = i - bwR
        ri = i + bwR + 1
        if li < 0: li = 0
        if ri > n: ri = n
        bw = ri - li
        Ri = <double*>PyArray_GETPTR2(R, li, i)
        # argMax = li + cblas_idamax(bw, Ri, n)
        # if R[argMax,i] < 0.:
        # assert R[argMax,i] == (Ri+cblas_idamax(bw, Ri, n)*n)[0]
        if (Ri+cblas_idamax(bw, Ri, n)*n)[0] < 0.:
            cblas_dscal(bw, -1, Ri, n) # invert signs

cdef inline void cSymmetrizeLowerToUpper(int n, double *A, int bwA):
    """
    Copy lower triangle of A to upper triangle of A.
    """
    cdef double *Ai, *Aj
    cdef int i
    cdef int inc = n+1
    Ai = A + n ; Aj = A + 1
    if bwA <= 0 or bwA >= n: 
        # for i in range(n-1, 0, -1):
        for i from n-1 >= i > 0:
            # A[i,i:] = A[i:,i]
            cblas_dcopy(i, Ai, n, Aj, 1)
            Ai += inc ; Aj += inc
    else:
        for i from n-1 >= i > bwA:
            cblas_dcopy(bwA, Ai, n, Aj, 1)
            Ai += inc ; Aj += inc
        for i from bwA >= i > 0:
            cblas_dcopy(i, Ai, n, Aj, 1)
            Ai += inc ; Aj += inc

cdef inline void cSymmetrizeUpperToLower(int n, double *A, int bwA):
    """
    Copy upper triangle of A to lower triangle of A.
    """
    cdef double *Ai, *Aj
    cdef int i
    cdef int inc = n+1
    Ai = A + n ; Aj = A + 1
    if bwA <= 0 or bwA >= n: 
        # for i in range(n-1, 0, -1):
        for i from n-1 >= i > 0:
            # A[i:,i] = A[i,i:]
            cblas_dcopy(i, Aj, 1, Ai, n)
            Ai += inc ; Aj += inc
    else:
        for i from n-1 >= i > bwA:
            cblas_dcopy(bwA, Aj, 1, Ai, n)
            Ai += inc ; Aj += inc
        for i from bwA >= i > 0:
            cblas_dcopy(i, Aj, 1, Ai, n)
            Ai += inc ; Aj += inc

def symmetrizeUpperToLower(N.ndarray A, int bwA = 0):
    cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] cA = A
    assert A.shape[0] == A.shape[1]
    cSymmetrizeUpperToLower(cA.shape[0], <double*>cA.data, bwA)

def symmetrizeLowerToUpper(N.ndarray A, int bwA = 0):
    cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] cA = A
    assert A.shape[0] == A.shape[1]
    cSymmetrizeLowerToUpper(cA.shape[0], <double*>cA.data, bwA)

cdef void cPermute(double *A0, N.ndarray permutation, N.ndarray C):
    """
    Permute the upper triangle of a matrix A, using temporary space C. 
    TODO: Could be be replaced by an efficient in-place permutation 
    algorithm
    """
    cdef unsigned int n = permutation.shape[0]
    cdef int i
    cdef double *Ai = A0
    cdef double *Ci = <double*>C.data
    # copy upper triangle of A to C
    for i from n >= i > 0:
        cblas_dcopy(i, Ai, 1, Ci, 1) # set row of C (with diagonal)
        Ai += 1 ; Ci += n
        cblas_dcopy(i-1, Ai, 1, Ci, n) # set column of C (without diagonal)
        Ai += n ; Ci += 1
    N.take(C, permutation, axis = 0, out = C)
    N.take(C, permutation, axis = 1, out = C)
    Ai = A0
    Ci = <double*>C.data
    # copy upper triangle of C back to upper triangle of A
    # and copy lower triangle of A to C 
    for i from n >= i > 0:
        cblas_dcopy(i, Ci, 1, Ai, 1) # copy rows back (with diagonal)
        Ai += n + 1 ; Ci += n + 1

@cython.cdivision(True)
cdef inline void cRtAR(int n, double *A, double *R, int bwR, double *C, ):
    """
    Similrity transformation (in place) of A using band matrix R and temporary
    matrix C.
    """
    if bwR <= 0 or bwR >= n // 3: # multiplication on full matrix
        # C = A*R
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    n, n, n, 1, A, n, R, n, 0, C, n)
        # A = R.T*C
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                    n, n, n, 1, R, n, C, n, 0, A, n)
        return 
    cdef int ii, jj, i, j, lj, li, ri, rj, k, l
    cdef double h
    ri = bwR + 1
    h = ri/-2.
    li = floor(h + sqrt(h*h + ELEMENTS_PER_BLOCK))
    if li > n: li = n
    elif li <= bwR: li = bwR+1
    k = n // li 
    if k > 2:
        # we get k blocks and rj remaining vectors which we add to 
        # the first block
        lj = n % li 
        lj += li
    else: 
        # but we need at least 2 blocks, because otherwise the 
        # algorithm won't work
        k = 2
        li = n//2
        lj = n-li
    rj = ri + lj
    if rj > n: rj = n
    # now the first block will be of dimension 
    # (li + n % li) x (bwR + 1 + li + n % li) = lj x rj and all 
    # others of dimension li x (bwR + 1 + li) = li x ri
    ri = 2*bwR + 1 + li # block size
    l = lj + (k-2)*li # upper bound for iteration over blocks

    ii = lj - bwR
    i = lj-li # initialize for the case of only two blocks
    for i in range(lj, l, li): 
    # for i from lj <= i < l by li:
        # left column blocks 
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    li, lj, rj, 1, 
                    A+i*n, n, R, n, 0, C+i*n, n)
        # upper row blocks
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    lj, li, ri, 1, 
                    A+ii, n, R+ii*n+i, n, 0, C+i, n)
        jj = lj - bwR
        for j in range(lj, l, li): 
        # for j from lj <= j < l by li:
            # middle part
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        li, li, ri, 1, 
                        A+i*n+jj, n, R+jj*n+j, n, 0, C+i*n+j, n)
            jj += li
        j += li
        # right row
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    li, li, n-j+bwR, 1, 
                    A+i*n+jj, n, R+jj*n+j, n, 0, C+i*n+j, n)
        ii += li
    i += li
    # lower row
    jj = lj - bwR
    for j in range(lj, l, li): 
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    li, li, ri, 1, 
                    A+i*n+jj, n, R+jj*n+j, n, 0, C+i*n+j, n)
        jj += li
    # upper left corner
    ri = n+bwR-i
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                lj, lj, rj, 1, 
                A, n, R, n, 0, C, n)
    # upper right corner 
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                lj, li, ri, 1, 
                A+ii, n, R+ii*n+i, n, 0, C+i, n)
    # lower left corner
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                li, lj, rj, 1, 
                A+i*n, n, R, n, 0, C+i*n, n)
    # lower right corner
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                li, li, ri, 1, 
                A+i*n+ii, n, R+ii*n+i, n, 0, C+i*n+i, n)


    # A = R.T*C <=> A.T = C.T*R = A since A = A.T
    ri = 2*bwR + 1 + li # reinitialize block size
    ii = lj - bwR
    i = lj-li # initialize for case that there are only two blocks
    for i in range(lj, l, li): 
        # USING SYMMETRY OF RESULTING A
        # # left column blocks 
        # cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
        #             li, lj, rj, 1, 
        #             C+i, n, R, n, 0, A+i*n, n)
        # upper row blocks
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                    lj, li, ri, 1, 
                    C+ii*n, n, R+ii*n+i, n, 0, A+i, n)
        # jj = lj - bwR
        jj = i - bwR
        for j in range(i, l, li): 
            # middle part
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                        li, li, ri, 1, 
                        C+jj*n+i, n, R+jj*n+j, n, 0, A+i*n+j, n)
            jj += li
        j += li
        # right column
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                    li, li, n-j+bwR, 1, 
                    C+jj*n+i, n, R+jj*n+j, n, 0, A+i*n+j, n)
        ii += li
    i += li
    # # lower row
    # jj = lj - bwR
    # for j in range(lj, l, li): 
    #     cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
    #                 li, li, ri, 1, 
    #                 C+jj*n+i, n, R+jj*n+j, n, 0, A+i*n+j, n)
    #     jj += li
    # upper left corner
    ri = n+bwR-i
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                lj, lj, rj, 1, 
                C, n, R, n, 0, A, n
                ) 
    # upper right corner 
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                lj, li, ri, 1, 
                C+ii*n, n, R+ii*n+i, n, 0, A+i, n)
    # # lower left corner
    # cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
    #             li, lj, rj, 1, 
    #             C+i, n, R, n, 0, A+i*n, n)
    # lower right corner
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                li, li, ri, 1, 
                C+ii*n+i, n, R+ii*n+i, n, 0, A+i*n+i, n)
    cSymmetrizeUpperToLower(n, A, 0)

def RtAR(N.ndarray A, N.ndarray R, int bwR = 0, N.ndarray C = None):
    """
    Similrity transformation (in place) of A using band matrix R and temporary
    matrix C.
    This is a wrapper for Pyhton, that calls the corresponding Cython function.
    """
    assert A.ndim == 2
    assert R.ndim == 2
    n = A.shape[0]
    assert A.shape[1] == n
    assert R.shape[0] == n
    assert R.shape[1] == n
    if bwR == 0: bwR = n - 1
    if C is None: C = N.empty((n,n), dtype = N.float)
    cRtAR(n, <double*>A.data, <double*>R.data, bwR, <double*>C.data)
    return A

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int cRotateR(int n, double *R, int bwR, 
        unsigned int i, unsigned int j, double c, double s, double tau):
    """
    Rotate R
    """
    cdef unsigned int k, l, ki, kj, pi, pj
    cdef int li, ri, lj, rj # boundaries of band
    cdef double g, h
    # do a rotation on full R if its bandwidth is large
    # i.e. if the bandwidth of R exceeds the maximal bandwidth
    if bwR > n // 3:
        ki = i ; kj = j
        for k in range(n):
            g = R[ki]
            h = R[kj]
            R[ki] -= s*(h + tau*g)
            R[kj] += s*(g - tau*h)
            ki += n ; kj += n
    # if not, the same five cases like for A exist.
    else:
        li = i - bwR
        ri = i + bwR + 1
        lj = j - bwR
        rj = j + bwR + 1
        if lj <= 0:
            li = lj = 0
            # assert ri <= n and rj <= n
            ki = i ; kj = j
            for k in range(ri):
                g = R[ki]
                h = R[kj]
                R[ki] -= s*(h + tau*g)
                R[kj] += s*(g - tau*h)
                ki += n ; kj += n
            g = tau * s
            ki = kj = ri*n
            ki += i ; kj += j
            for k in range(ri, rj):
                R[ki] -= s*R[kj]
                R[kj] -= g*R[kj]
                ki += n ; kj += n
        elif li < 0:
            li = 0
            # assert ri <= n and rj <= n
            ki = kj = lj*n
            ki += i ; kj += j
            for k in range(lj, ri):
                g = R[ki]
                h = R[kj]
                R[ki] -= s*(h + tau*g)
                R[kj] += s*(g - tau*h)
                ki += n ; kj += n
            g = tau * s
            ki = i ; kj = j
            for k in range(lj):
                R[kj] += s*R[ki]
                R[ki] -= g*R[ki]
                ki += n ; kj += n
            ki = kj = ri*n
            ki += i ; kj += j
            for k in range(ri, rj):
                R[ki] -= s*R[kj]
                R[kj] -= g*R[kj]
                ki += n ; kj += n
        elif ri >= n:
            ri = rj = n
            # assert li >= 0 and lj >= 0
            ki = kj = lj*n
            ki += i ; kj += j
            for k in range(lj, n):
                g = R[ki]
                h = R[kj]
                R[ki] -= s*(h + tau*g)
                R[kj] += s*(g - tau*h)
                ki += n ; kj += n
            g = tau * s
            ki = kj = li*n
            ki += i ; kj += j
            for k in range(li, lj):
                R[kj] += s*R[ki]
                R[ki] -= g*R[ki]
                ki += n ; kj += n
        elif rj > n: 
            rj = n
            # assert li >= 0 and lj >= 0
            ki = kj = lj*n
            ki += i ; kj += j
            for k in range(lj, ri):
                g = R[ki]
                h = R[kj]
                R[ki] -= s*(h + tau*g)
                R[kj] += s*(g - tau*h)
                ki += n ; kj += n
            g = tau * s
            ki = kj = li*n
            ki += i ; kj += j
            for k in range(li, lj):
                R[kj] += s*R[ki]
                R[ki] -= g*R[ki]
                ki += n ; kj += n
            ki = kj = ri*n
            ki += i ; kj += j
            for k in range(ri, n):
                R[ki] -= s*R[kj]
                R[kj] -= g*R[kj]
                ki += n ; kj += n
        else:
            # assert li >= 0 and lj >= 0
            # assert ri <= n and rj <= n
            ki = kj = lj*n
            ki += i ; kj += j
            for k in range(lj, ri):
                g = R[ki]
                h = R[kj]
                R[ki] -= s*(h + tau*g)
                R[kj] += s*(g - tau*h)
                ki += n ; kj += n
            g = tau * s
            # assert lj - li == rj - ri
            ki = kj = li*n
            ki += i ; kj += j
            pi = pj = ri*n
            pi += i ; pj += j
            for k in range(li, lj):
                R[kj] += s*R[ki]
                R[ki] -= g*R[ki]
                R[pi] -= s*R[pj]
                R[pj] -= g*R[pj]
                ki += n ; kj += n
                pi += n ; pj += n
        # now check if the bandwidth of R gets larger by 
        # this rotation
        if ri < rj:
            ki = (rj-1)*n + i
            for k in range(rj-1, ri-2, -1):
            # for k from rj-1 >= k > ri-2:
                if fabs(R[ki]) >= ZERO: break
                ki -= n
            # use l as temporary variable to store the value 
            # that has to be added to the diagonal
            l = k - ri + 1
        else:
            l = 0
        if li < lj:
            kj = li*n + j
            for k in range(li, lj+1):
                if fabs(R[kj]) >= ZERO: break
                kj += n
            # check if bandwidth is larger in this direction
            # i.e. lj - k > k - ri + 1
            lj -= k
            if lj > l: l = lj
        bwR += l # update bandwidth of R
    return bwR


cdef class JacobiBandDiagonalization:
    """
    This class provides a diagonalization procedure for real symmetric 
    band matrices (in full format only at the moment).
    """
    cdef readonly N.ndarray A
    cdef readonly N.ndarray d
    cdef readonly N.ndarray R
    cdef int n
    cdef readonly int bwA, bwR, bwMax
    cdef readonly N.ndarray permutation
    cdef readonly double threshold
    cdef StorageInfo A0available
    cdef FinallyDo finallyDo
    cdef readonly double error
    # cdef public N.ndarray check

    def getA0Info(self, type = str):
        if type == str:   return StorageInfoDict[int(<int>self.A0available)]
        elif type == int: return int(<int>self.A0available)
        else:             return self.A0available

    def __len__(self):
        return self.n

    # @cython.boundscheck(False)
    def __init__(self, N.ndarray[N.float_t, ndim = 2, mode = 'c'] A not None, 
                       N.ndarray[N.float_t, ndim = 2, mode = 'c'] R = None, 
                       unsigned int bandwidth = 0, unsigned int bwR = 0, 
                       ):
        """
        A is a real symmetric band matrix (as numpy ndarray). The bandwidth (bw)
        is defines the band within the diagonalization should be performed.
        The rotation matrix R will be the eigenvector matrix and will be 
        initialized as identity matrix if R is None. A matrix can be given
        if A is already prediagonalized.
        If the bandwidth of the n times n matrix A is larger than n devided by
        three, A will be treated as full matrix and thus the algorithm won't 
        be very efficient. Set bandwidth to 0 to diagonalize the full matrix.
        """
        # self.A = A.copy() # we will work on a copy
        self.A = A
        self.d = N.diag(A)
        self.n = n = len(A) 
        cdef int bwA = bandwidth # bandwidth of A
        cdef int bwMax = n // 3 # bwMax is maximal bandwidth
        # do some checks on the bandwidth
        # if bwA > bwMax:
        #     bwA = bwMax
        #     print("Setting bandwidth to maximum of %i." %bwMax)
        if bwA >= n:
            bwA = n-1
            print("Setting bandwidth to maximum of %i." %bwA)
        elif bwA == 0:
            bwA = n-1 # do diagonalization on full matrix
        self.bwA = bwA
        self.bwMax = bwMax
        cdef int i, j
        if R is None:
            self.R = N.identity(n, dtype = A.dtype)
            self.A0available = LOWER_TRIANGLE
        else:
            assert A.shape[0] == R.shape[0]
            assert A.shape[1] == R.shape[1]
            self.R = R
            # determine bandwidth if it has not been given
            if bwR == 0: bwR = cGetBandwidth(R)
            self.A0available = NOT_AVAILABLE
        if bwR < bwA:
            bwR = bwA
        else:
            bwR = bwR
        self.bwR = bwR

    def setupSweep(self):
        """
        Overwrite this method to do some setup stuff for each sweep.
        """
        return 

    def getError(self):
        """
        Error: sum_(i<j) |A_ij|
        """
        A = self.A
        self.error =  N.sum(N.triu(N.power(N.abs(A), 2), k=1))
        return self.error

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double threshold = 1.0e-15, unsigned int maxIter = 100, 
                       printInfo = False, sort = False, finallyDo = 'trafo', 
                       signConvention = True):
        """
        Call, to start the diagonalization procedure (in place!).
        The parameters are:
            'threshold'     defines the maximal magnitude for the off-diagonal 
                            elements inside the defined band. 
            'maxIter'       maximal number of iterations (sweeps).
            'printInfo'     print some infos.
            'sort'          sort the matrices A and R according to the diagonal 
                            of A (eigenvalues).
                            WARNING: Energetic sorting may influence the 
                                     bandwidths of A and R which may reduce 
                                     efficiency, if the diagonalization 
                                     process is restarted after sorting.
                                     (Sorting will be omitted if the 
                                     diagonalization won't converge, so 
                                     restarting is save in this case.)
            'finallyDo'     if 'finallyDo' is 'TRAFO', 'T' or 1,  the 
                            returned (partially) diagonalized matrix will be 
                            returned as R.T*A0*R. 
                            To save time, the step can be omitted by setting
                            'finallyDo' to 'FILL', 'F', or 2. Then A will 
                            be the matrix resulting from the rotations, only
                            a copy step is performed where the upper triangle
                            is copied to the lower triangle. 
                            Note: One has to take care, that there are no 
                                  significant elements outside the band of 
                                  A and that they also cannot appear during 
                                  the diagonalization process, since
                                  A is not updated outside the band during the 
                                  diagonalization. 
                            'finallyDo' may also set to 0 (or anything else)
                            to keep the matrix as it is, i.e. the original 
                            matrix A0 is stored on the lower triangle and 
                            the rotated A is stored on the upper triangle.
                            The diagonal of the returned A contains the 
                            elements of the rotated matrix. (The diagonal
                            elements of A0 are stored in 'self.d'.)
        NOTE: The diagonalization itself is an O(n*bwA^2*bwR) process, but 
              the final evaluation of R.T*A0*R is an O(n^2*bwR) process, 
              which can be omitted if A is a strict band matrix and the 
              selected bandwidth is sufficiently large for the diagonalization 
              process (where is len(A) and bwA and bwR are the bandwidths 
              of A and R respectively).

        To understand the algorithm, take a look at the following picture and 
        note:
        d are the diagonal elements and o the off-diagonal elements. One
        rotation will be applied on rows and columns i an j, were we have to 
        make sure that i < j => li < lj and ri < rj. li and lj denote the left
        boundaries of the band.
        The rotations will operate only on one triangle (and the diagonal) of
        the matrix. All elements of A outside the band won't be updated durring
        the diagonalization procedure. In the end, the resulting R will 
        be applied on the original matrix A0.
            |d(0,0) o(0,1) o(0,2) ... o(0,bw+1) 0             ...         0  |
            |o(1,0) d(1,1) o(1,2) o(1,3) ... o(1,bw+2) 1      ...         0  |
            |  0                      ...                                 0  |
            |  0    o(i,li) ... d(i,i) ... o(i,ri) ...                    0  |
        A = |  0                      ...                                 0  |
            |  0                  o(j,lj) ... d(j,j) ... o(j,rj) ...      0  |
            |  0                      ...                                 0  |
            |  0    ...       o(n-1,n-1-bw ... o(n-1,n-2) d(n-1,n-1) o(n-1,n)|
            |  0        ...            o(n,n-bw) ... o(n,n-2) o(n,n-1) d(n,n)|
        """
        if isinstance(finallyDo, str): finallyDo = finallyDo.upper()

        if finallyDo in ('D', 'FILL_DIAGONAL', 0): self.finallyDo =FILL_DIAGONAL
        elif finallyDo in ('T', 'TRAFO', 1): self.finallyDo = TRAFO
        elif finallyDo in ('F', 'FILL', 2): self.finallyDo = FILL
        else: self.finallyDo = FILL_DIAGONAL # DEFAULT

        if self.finallyDo == TRAFO and self.A0available == NOT_AVAILABLE:
            raise AttributeError(
                "Final transformation is not possible, since the original " + 
                "matrix A0 is not available.")

        # A should be diagonalized ...
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] A = self.A 
        # ...by rotation matrix R
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] R = self.R 
        cdef unsigned int sweep, index, i, j, p, q, k, l # for iteration
        cdef int li, ri, lj, rj # boundaries of band
        cdef double theta, t, tau, g, h, c, s # some temporary variables
        cdef int n = self.n # dimension of A and R
        cdef int bwA = self.bwA # bwA is bandwidth A
        cdef int bwR = self.bwR # bwR is bandwidth R
        cdef int bwMax = self.bwMax # maximal bandwidth of A and R
        assert bwA >= 1

        # diagonal of A
        cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] d = self.d 

        # copy d (which may be the diagonal elements of A0) to the diagonal of A
        cblas_dswap(n, <double*>d.data, 1, <double*>A.data, n+1)

        # two arrays to stabilize diagonal elements 
        # (see Numerical Recipes in C p.467f)
        cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] b = d.copy()
        cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] z = N.zeros(n, 
                                                             dtype = N.float)
        # this array will be sorted to get the elements to be rotated 
        # in correct order
        cdef N.ndarray[SortType_int_int, ndim = 1] sortMe = N.empty( n, 
                           dtype = N.dtype( [('a', N.int), ('i', N.int)], 
                           align = True))
        # and we will iterate over this one later
        cdef N.ndarray[N.int_t, ndim = 1] indices

        cdef unsigned int converged = 0
        cdef unsigned int iter = 0 # sweep counter
        setupSweep = self.setupSweep
        for sweep in range(maxIter):
            iter += 1 # count sweeps
            converged = 1 # assume convergence 
            setupSweep()
            for q in range(1, bwA+1): # iterate over all subdiagonals in band
                # and have a look at all elements of this subdiagonal
                p = 0 # count the elements to be rotated 
                for k in range(n-q): # iterate over all elements in subdiagonal
                    g = fabs(A[k, k+q])
                    if g >= threshold:
                        # only sort by exponent, because sorting integers is
                        # faster and sufficiently accurate
                        sortMe[p].a = -logb(g) 
                        sortMe[p].i = k # and their index
                        p += 1
                # sort all elements larger than threshold, largest element of A
                # on the subdiagonal will be first
                sortMe[:p].sort(order = ['a',]) 
                indices = sortMe['i'][:p] # create a view of the sorted indices
                for index in range(p): # now get all sorted indices 
                    # we have to rotate again, as long as range(p) is not empty
                    # i.e. no convergence at the moment
                    converged = 0
                    i = indices[index]
                    j = i + q

                    # compute rotation angle theta 
                    # (see Numerical Recipes in C p.467f)
                    g = fabs(A[i,j])*100.0
                    # h = A[j,j] - A[i,i]
                    h = d[j] - d[i]
                    # detect overflow
                    if ( <double>(fabs(d[i]) + g) == <double>fabs(d[i]) and
                         <double>(fabs(d[j]) + g) == <double>fabs(d[j]) ):
                    # if ( <double>(fabs(A[i,i]) + g) == <double>fabs(A[i,i]) and
                    #      <double>(fabs(A[j,j]) + g) == <double>fabs(A[j,j]) ):
                        A[i,j] = 0.0
                        continue # skip the rotation
                    elif <double>(fabs(h) + g) == <double>fabs(h): 
                        t = A[i,j] / h
                    else:
                        theta = 0.5 * h / A[i,j]
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta*theta))
                        if theta < 0.0:
                            t = -t
                        # assert fabs(theta  - (0.5 * h / A[i,j]) ) < 1e-15
                    c = 1.0/sqrt(1+t*t)
                    s = t*c
                    tau = s / (1.0 + c)
                    h = t*A[i,j]
                    # get boundaries on rows (columns) i and j
                    li = i - bwA
                    ri = i + bwA + 1
                    lj = j - bwA
                    rj = j + bwA + 1
                    # assert not (li < 0 and ri > n)
                    # assert not (lj < 0 and rj > n)
                    # rotations for the following matrix elements are always 
                    # the same
                    # A[i,i] -= h
                    # A[j,j] += h
                    d[i] -= h ; d[j] += h
                    z[i] -= h ; z[j] += h
                    A[i,j] = 0.
                    # A[i,j] = (c*c - s*s)*A[i,j] + s*c*(A[i,i]-A[j,j])
                    # assert fabs((c*c-s*s)*A[i,j]+s*c*(A[i,i]-A[j,j]))<1.0e-15
                    # A[i,j] = A[j,i] = 0.
                    for k in range(i+1,j):
                        g = A[i,k]
                        h = A[k,j]
                        A[i,k] -= s*(h + tau*g)
                        A[k,j] += s*(g - tau*h)
                    # do calculations on full A, if bwA is large
                    if bwA > bwMax:
                        for k in range(i):
                            g = A[k,i]
                            h = A[k,j]
                            A[k,i] -= s*(h + tau*g)
                            A[k,j] += s*(g - tau*h)
                        for k in range(j+1, n):
                            g = A[i,k]
                            h = A[j,k]
                            A[i,k] -= s*(h + tau*g)
                            A[j,k] += s*(g - tau*h)
                    # now we have to distinguish between five different cases
                    # depending on the rotated element A[i,j]
                    # since it may lie a corner of the matrix (li = lj = 0 or 
                    # ri = rj = n) or close to a corner (li = 0, lj != 0) or 
                    # ri != n, rj = n) or somewhere else. Since the 
                    # maximal bandwidth is limited to n/3 there won't be a 
                    # case where li = 0 and rj = n at the same time (and also
                    # because i < j not li = 0 and ri = n and not lj = 0 rj = n)
                    elif lj <= 0: # highest corner of matrix (very small i,j)
                        # assert li < 0
                        # li = lj = 0 # debugging only
                        for k in range(i):
                            g = A[k,i]
                            h = A[k,j]
                            A[k,i] -= s*(h + tau*g)
                            A[k,j] += s*(g - tau*h)
                        for k in range(j+1, ri):
                            g = A[i,k]
                            h = A[j,k]
                            A[i,k] -= s*(h + tau*g)
                            A[j,k] += s*(g - tau*h)
                        g = tau * s
                        for k in range(ri, rj):
                            # A[i,k] -= g*A[j,k] # out of band update (DEBUGGING)
                            A[j,k] -= g*A[j,k]
                    elif li < 0: # higher corner of matrix (small i,j)
                        # assert lj >= 0
                        # li = 0 # debugging only
                        # assert i-lj == ri-(j+1)
                        l = j + 1
                        for k in range(lj, i):
                            g = A[k,i]
                            h = A[k,j]
                            A[k,i] -= s*(h + tau*g)
                            A[k,j] += s*(g - tau*h)
                            g = A[i,l]
                            h = A[j,l]
                            A[i,l] -= s*(h + tau*g)
                            A[j,l] += s*(g - tau*h)
                            l += 1
                        g = tau * s
                        for k in range(lj):
                            # A[k,j] += s*A[k,i] # out of band update (DEBUGGING)
                            A[k,i] -= g*A[k,i]
                        for k in range(ri, rj):
                            # A[i,k] -= s*A[j,k] # out of band update (DEBUGGING)
                            A[j,k] -= g*A[j,k]
                    elif ri >= n: # lowest corner of matrix (very large i,j)
                        # assert rj > n # if ri > n also rj > n since ri > rj
                        # ri = rj = n # debugging only
                        for k in range(lj, i):
                            g = A[k,i]
                            h = A[k,j]
                            A[k,i] -= s*(h + tau*g)
                            A[k,j] += s*(g - tau*h)
                        for k in range(j+1, n):
                            g = A[i,k]
                            h = A[j,k]
                            A[i,k] -= s*(h + tau*g)
                            A[j,k] += s*(g - tau*h)
                        g = tau * s
                        for k in range(li, lj):
                            # A[k,j] += s*A[k,i] # out of band update (DEBUGGING)
                            A[k,i] -= g*A[k,i]
                    elif rj > n: # lower corner of matrix (large i,j)
                        # assert ri <= n # only rj > n
                        # rj = n # debugging only
                        l = j + 1
                        for k in range(lj, i):
                            g = A[k,i]
                            h = A[k,j]
                            A[k,i] -= s*(h + tau*g)
                            A[k,j] += s*(g - tau*h)
                            g = A[i,l]
                            h = A[j,l]
                            A[i,l] -= s*(h + tau*g)
                            A[j,l] += s*(g - tau*h)
                            l += 1
                        g = tau * s
                        for k in range(li, lj):
                            # A[k,j] += s*A[k,i] # out of band update (DEBUGGING)
                            A[k,i] -= g*A[k,i]
                        for k in range(ri, n):
                            # A[i,k] -= s*A[j,k] # out of band update (DEBUGGING)
                            A[j,k] -= g*A[j,k]
                    else: # middle part of matrix 
                        # i < j => li < lj
                        # assert i-lj == ri-(j+1)
                        # assert lj-li == rj-ri
                        l = j + 1
                        for k in range(lj, i):
                            g = A[k,i]
                            h = A[k,j]
                            A[k,i] -= s*(h + tau*g)
                            A[k,j] += s*(g - tau*h)
                            g = A[i,l]
                            h = A[j,l]
                            A[i,l] -= s*(h + tau*g)
                            A[j,l] += s*(g - tau*h)
                            l += 1
                        g = tau * s
                        l = ri
                        for k in range(li, lj):
                            # A[k,j] += s*A[k,i] # out of band update (DEBUGGING)
                            A[k,i] -= g*A[k,i]
                            # A[i,l] -= s*A[j,l] # out of band update (DEBUGGING)
                            A[j,l] -= g*A[j,l]
                            l += 1
                    # now R has to be updated
                    bwR = cRotateR(n, <double*>R.data, bwR, i, j, c, s, tau)
            # stabilize diagonal elements (see Numerical Recipes in C p.467f)
            for i in range(n):
                b[i] += z[i]
                # A[i,i] = b[i]
                d[i] = b[i]
                z[i] = 0.0
            if converged == 1: 
                break
        self.d = d # store eigenvalues

        # update the bandwidth of R
        if bwR >= bwMax:
            bwR = cGetBandwidth(R)
        self.bwR = bwR
        # assert RisIdentity(R)

        # renormalize vectors of R
        for i in range(n):
            li = i - bwR
            if li < 0: li = 0
            ri = i + bwR + 1
            if ri > n: ri = n
            g = 1./cblas_dnrm2(ri-li, <double*>PyArray_GETPTR2(R, li, i), n) 
            if fabs(1.0 - g) > ZERO:
                # R[li:ri,i] *= g
                cblas_dscal(ri-li, g, <double*>PyArray_GETPTR2(R, li, i), n)

        if iter >= maxIter and converged == 0:
            if self.A0available != NOT_AVAILABLE or self.finallyDo != TRAFO:
                cblas_dswap(n, <double*>d.data, 1, <double*>A.data, n+1)
                raise ValueError('No convergence after %i steps !' %iter + 
                                    '(Restart possible)')
            else:
                raise ValueError('No convergence after %i steps !' %iter + 
                                    '(Restart impossible)')
        if printInfo:
            print("Needed %i iterations." %iter) 

        # make largest element of each eigenvector positive
        if signConvention: cSetSignsOfEigenvectors(R, bwR) 

        # either return A directly or calculate R.T*A*R
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] C = N.empty( 
                                                    (n,n), dtype = N.float)
        if self.finallyDo == TRAFO:
            # restore A0 from lower triangle of A
            if self.A0available == LOWER_TRIANGLE:
                cSymmetrizeLowerToUpper(n, <double*>A.data, bwA)
            # restore A0 from upper triangle of A
            elif self.A0available == UPPER_TRIANGLE:
                cSymmetrizeUpperToLower(n, <double*>A.data, bwA)
            else: raise AssertionError('Something went terribly wrong!')
            # Now calculate the accurate matrix resulting from the rotations
            cRtAR(n, <double*>A.data, # to be rotated (in place)
                    <double*>R.data, bwR, # rotation matrix
                    <double*>C.data,  # temporary array
                    )
            # finally set A0available to NOT_AVAILABLE, 
            # since we cannot do a final transformation again, since we lost A0.
            self.A0available = NOT_AVAILABLE
        # No final similarity transformation, just fill A 
        elif self.finallyDo == FILL:
            cblas_dswap(n, <double*>d.data, 1, <double*>A.data, n+1)
            # build full A from upper triangle and 'd'
            if self.A0available == LOWER_TRIANGLE:
                cSymmetrizeUpperToLower(n, <double*>A.data, bwA)
            elif self.A0available == UPPER_TRIANGLE:
                cSymmetrizeLowerToUpper(n, <double*>A.data, bwA)
            else: raise AssertionError('Something went terribly wrong!')
            self.A0available = NOT_AVAILABLE
        # only fill in diagonal and store diagonal of A0 in d
        elif self.finallyDo == FILL_DIAGONAL:
            cblas_dswap(n, <double*>d.data, 1, <double*>A.data, n+1)
        else: raise AssertionError('Something went terribly wrong!')

        if sort: self._sort() # sort eigenvectors and permute A
        return (A, R) # return eigenvectors as columns

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void _sort(self, sortA = True, bwRupdate = True):
        cdef unsigned int n = self.n
        cdef int i
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] A = self.A 
        cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] d = self.d
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] R = self.R
        cdef N.ndarray[SortType_float_int, ndim = 1] sortMe = N.empty(n, 
                           dtype = N.dtype( [('a', N.float), ('i', N.int)], 
                           align = True))
        sortMe['a'] = N.diag(A)
        # sortMe['a'] = d.copy() # we have diag(A0) in self.d now
        sortMe['i'] = N.arange(n, dtype = N.int)
        sortMe.sort(order = ['a',])
        cdef N.ndarray[N.int_t, ndim = 1] permutation = sortMe['i']
        self.permutation = permutation
        # N.take(d, permutation, axis = 0, out = d) # we don't order diag of A0
        N.take(R, permutation, axis = 1, out = R)

        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] C 
        if self.A0available == NOT_AVAILABLE:
            N.take(A, permutation, axis = 0, out = A) 
            N.take(A, permutation, axis = 1, out = A)
        elif self.A0available == LOWER_TRIANGLE:
            # diagonal of A0 is in self.d, so diagonal of A must be permuted
            C = N.empty((n, n), dtype = N.float)
            cPermute(<double*>A.data, permutation, C)
        elif self.A0available == UPPER_TRIANGLE:
            raise AssertionError("Not implemented!")
        if bwRupdate: self.bwR = cGetBandwidth(R)

cdef class SimultaneousJacobiBandDiagonalization(JacobiBandDiagonalization):
    """
    This class provides a simultaneous diagonalization procedure for real 
    symmetric band matrices (in full format only at the moment).
    """
    cdef public N.ndarray a, b
    cdef public double ab
    cdef public int m # we diagonalize m matrices simultaneously

    # @cython.boundscheck(False)
    def __init__(self, N.ndarray[N.float_t, ndim = 3, mode = 'c'] A not None, 
                       N.ndarray[N.float_t, ndim = 2, mode = 'c'] R = None, 
                       unsigned int bandwidth = 0, unsigned int bwR = 0
                       ):
        """
        A is an array of real symmetric band matrices (as numpy ndarray). 
        The bandwidth (bw) is defines the band within the diagonalization 
        should be performed.
        """
        cdef int i, j, n, m
        m = A.shape[0] 
        n = A.shape[1]
        assert A.shape[1] == A.shape[2]
        self.m = m # we diagonalize m matrices simultaneously
        self.a = N.empty(m, dtype = N.float)
        self.b = N.empty(m, dtype = N.float)
        JacobiBandDiagonalization.__init__(self, A[0], R, bandwidth, bwR)
        self.A = A
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] d = N.empty( (m, n), 
                                                                dtype = N.float)
        cdef double *Ai = <double*>A.data
        cdef double *di = <double*>d.data
        cdef int nn = n*n
        cdef int nPlusOne = n+1
        for i in range(m):
            cblas_dcopy(n, Ai, nPlusOne, di, 1) 
            Ai += nn ; di += n
        self.d = d

    def diagG(self, int i, int j):
        """
        Calculate diagonal element of G-Matrix of eq. (4) in J.-F. Cardoso 
        and A. Souloumiac, SIAM J. Mat. Anal. Appl. 17 (1995) pp. 161-164.
        (May be overwritten by a subclass to include some mappings on the 
        different matrices.)
        """
        cdef double *a = <double*>self.a.data
        cdef double *b = <double*>self.b.data
        cdef int m = self.m
        cdef double ab = self.ab
        ab *= 2 # 2*G12
        ab *= ab # 4*G12^2
        cdef double aa = cblas_ddot(m, a, 1, a, 1) # N.dot(a, a) # G11
        cdef double bb = cblas_ddot(m, b, 1, b, 1) # N.dot(b, b) # G22
        aa -= bb # G11 - G22
        cdef double x
        x = aa      # x = (G11 - G22)
        x *= x      # x = (G11 - G22)^2
        x += ab     # x = (G11 - G22)^2 + 4*G12^2
        x = sqrt(x) # x = sqrt( (G11 - G22)^2 + 4*G12^2 ) 
        x += aa     # x = G11 - G22 + sqrt( (G11 - G22)^2 + 4*G12^2 ) 
        return x

    def offDiagG(self, int i, int j):
        """
        Calculate off-diagonal element of G-Matrix of eq. (4) in J.-F. Cardoso 
        and A. Souloumiac, SIAM J. Mat. Anal. Appl. 17 (1995) pp. 161-164.
        (May be overwritten by a subclass to include some mappings on the 
        different matrices.)
        """
        cdef N.ndarray[N.float_t, ndim = 3, mode = 'c'] A = self.A
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] d = self.d
        cdef int m = self.m
        cdef int n = self.n
        cdef double* a = <double*>self.a.data
        cdef double* b = <double*>self.b.data
        # a = N.subtract(d[:,j], d[:,i], a)
        cblas_dcopy(m, <double*>PyArray_GETPTR2(d, 0, j), n, a, 1)
        cblas_daxpy(m, -1, <double*>PyArray_GETPTR2(d, 0, i), n, a, 1)
        # b = N.multiply(A[:,i,j], 2, b)
        cblas_dcopy(m, <double*>PyArray_GETPTR3(A, 0, i, j), n*n, b, 1)
        cblas_dscal(m, 2, b, 1)
        # ab = N.dot(a, b)
        cdef double ab = cblas_ddot(m, a, 1, b, 1)
        self.ab = ab
        return ab

    def getError(self):
        """
        Error: sum_(i<j) sum_k |A_k_ij|^2
        """
        A = self.A
        m = self.m
        self.error = N.sum(N.triu(N.sum(N.power(N.abs(A[:m]),2),axis = 0),k=1))
        return self.error

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double threshold = 1.0e-15, unsigned int maxIter = 100, 
                       printInfo = False, sort = False, finallyDo = 'trafo', 
                       signConvention = True):
        """
        Call, to start the diagonalization procedure (in place!).
        (See also documentation of parent class JacobiBandDiagonalization.)
        """
        if isinstance(finallyDo, str): finallyDo = finallyDo.upper()

        if finallyDo in ('D', 'FILL_DIAGONAL', 0): self.finallyDo =FILL_DIAGONAL
        elif finallyDo in ('T', 'TRAFO', 1): self.finallyDo = TRAFO
        elif finallyDo in ('F', 'FILL', 2): self.finallyDo = FILL
        else: self.finallyDo = FILL_DIAGONAL # DEFAULT

        if self.finallyDo == TRAFO and self.A0available == NOT_AVAILABLE:
            raise AttributeError(
                "Final transformation is not possible, since the original " + 
                "matrix A0 is not available.")


        # A should be diagonalized ...
        cdef N.ndarray[N.float_t, ndim = 3, mode = 'c'] A = self.A 
        # ...by rotation matrix R
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] R = self.R 
        cdef unsigned int sweep, index, i, j, p, q, k, l, f # for iteration
        cdef int li, ri, lj, rj # boundaries of band
        cdef double t, theta, tau, g, h, c, s # some temporary variables
        cdef int n = self.n # dimension of A and R
        cdef int m = self.m # number of matrices to be diagonalized
        cdef int bwA = self.bwA # bwA is bandwidth A
        cdef int bwR = self.bwR # bwR is bandwidth R
        cdef int bwMax = self.bwMax # maximal bandwidth of A and R

        cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] v = N.empty(m, 
                                                                dtype = N.float)
        # diagonal of A
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] d = self.d 

        # copy d (which may be the diagonal elements of A0) to the diagonal of A
        for f in range(m):
            cblas_dswap(n, <double*>PyArray_GETPTR2(d, f, 0), 1, 
                           <double*>PyArray_GETPTR3(A, f, 0, 0), n+1)

        # two arrays to stabilize diagonal elements 
        # (see Numerical Recipes in C p.467f)
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] b = N.array(d, 
                                                copy = True, dtype = N.float)
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] z = N.zeros( (m, n), 
                                                             dtype = N.float)

        # this array will be sorted to get the elements to be rotated 
        # in correct order
        cdef N.ndarray[SortType_int_int, ndim = 1] sortMe = N.empty( n, 
                           dtype = N.dtype( [('a', N.int), ('i', N.int)], 
                           align = True))
        # and we will iterate over this one later
        cdef N.ndarray[N.int_t, ndim = 1] indices

        cdef unsigned int converged = 0
        cdef unsigned int iter = 0 # sweep counter
        offDiagG = self.offDiagG
        diagG = self.diagG
        setupSweep = self.setupSweep
        for sweep in range(maxIter):
            iter += 1 # count sweeps
            converged = 1 # assume convergence 
            setupSweep()
            for q in range(1, bwA+1): # iterate over all subdiagonals in band
                # and have a look at all elements of this subdiagonal
                p = 0 # count the elements to be rotated 
                for k in range(n-q): # iterate over all elements in subdiagonal
                    # g = fabs(N.sum(A[:,k,k+q]))
                    g = fabs(offDiagG(k, k+q))
                    if g >= threshold:
                        # only sort by exponent, because sorting integers is
                        # faster and sufficiently accurate
                        sortMe[p].a = -logb(g) 
                        sortMe[p].i = k # and their index
                        p += 1
                # sort all elements larger than threshold, largest element of A
                # on the subdiagonal will be first
                sortMe[:p].sort(order = ['a',]) 
                indices = sortMe['i'][:p] # create a view of the sorted indices
                for index in range(p): # now get all sorted indices 
                    # we have to rotate again, as long as range(p) is not empty
                    # i.e. no convergence at the moment
                    converged = 0
                    i = indices[index]
                    j = i + q

                    g = offDiagG(i, j) 
                    h = diagG(i, j)

                    if <double>(fabs(h) + fabs(g*100)) == <double>fabs(h): 
                        t = g / h
                    else:
                        theta = 0.5 * h / g
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta*theta))
                        if theta < 0.0:
                            t = -t
                    c = 1.0/sqrt(1+t*t)
                    s = t*c
                    tau = s / (1.0 + c)

                    # v = N.multiply(A[:,i,j], t, v) 
                    cblas_dcopy(m, <double*>PyArray_GETPTR3(A, 0, i, j), n*n,
                                   <double*>v.data, 1)
                    cblas_dscal(m, t, <double*>v.data, 1)

                    # get boundaries on rows (columns) i and j
                    li = i - bwA
                    ri = i + bwA + 1
                    lj = j - bwA
                    rj = j + bwA + 1
                    # assert not (li < 0 and ri > n)
                    # assert not (lj < 0 and rj > n)
                    # rotations for the following matrix elements are always 
                    # the same
                    for f in range(m):
                        # A[f,i,j] = (c*c - s*s)*A[f,i,j] + s*c*(d[f,i]-d[f,j])
                        A[f,i,j] = g = (c*c-s*s)*A[f,i,j] + s*c*(d[f,i]-d[f,j])
                        # h = v[f]
                        h = v[f] + g*t
                        d[f,i] -= h ; d[f,j] += h
                        z[f,i] -= h ; z[f,j] += h
                        for k in range(i+1,j):
                            g = A[f,i,k]
                            h = A[f,k,j]
                            A[f,i,k] -= s*(h + tau*g) 
                            A[f,k,j] += s*(g - tau*h)
                    # do calculations on full A, if bwA is large
                    if bwA > bwMax:
                        for f in range(m):
                            for k in range(i):
                                g = A[f,k,i]
                                h = A[f,k,j]
                                A[f,k,i] -= s*(h + tau*g)
                                A[f,k,j] += s*(g - tau*h)
                            for k in range(j+1, n):
                                g = A[f,i,k]
                                h = A[f,j,k]
                                A[f,i,k] -= s*(h + tau*g)
                                A[f,j,k] += s*(g - tau*h)
                    # now we have to distinguish between five different cases
                    # depending on the rotated element A[i,j]
                    # since it may lie a corner of the matrix (li = lj = 0 or 
                    # ri = rj = n) or close to a corner (li = 0, lj != 0) or 
                    # ri != n, rj = n) or somewhere else. Since the 
                    # maximal bandwidth is limited to n/3 there won't be a 
                    # case where li = 0 and rj = n at the same time (and also
                    # because i < j not li = 0 and ri = n and not lj = 0 rj = n)
                    elif lj <= 0: # highest corner of matrix (very small i,j)
                        # self.check[i,j] = 2
                        # assert li < 0
                        # li = lj = 0 # debugging only
                        for f in range(m):
                            for k in range(i):
                                g = A[f,k,i]
                                h = A[f,k,j]
                                A[f,k,i] -= s*(h + tau*g)
                                A[f,k,j] += s*(g - tau*h)
                            for k in range(j+1, ri):
                                g = A[f,i,k]
                                h = A[f,j,k]
                                A[f,i,k] -= s*(h + tau*g)
                                A[f,j,k] += s*(g - tau*h)
                            g = tau * s
                            for k in range(ri, rj):
                                # A[f,i,k] -= g*A[f,j,k] # out of band update (DEBUGGING)
                                A[f,j,k] -= g*A[f,j,k]
                    elif li < 0: # higher corner of matrix (small i,j)
                        # assert lj >= 0
                        # li = 0 # debugging only
                        # assert i-lj == ri-(j+1)
                        for f in range(m):
                            l = j + 1
                            for k in range(lj, i):
                                g = A[f,k,i]
                                h = A[f,k,j]
                                A[f,k,i] -= s*(h + tau*g)
                                A[f,k,j] += s*(g - tau*h)
                                g = A[f,i,l]
                                h = A[f,j,l]
                                A[f,i,l] -= s*(h + tau*g)
                                A[f,j,l] += s*(g - tau*h)
                                l += 1
                            g = tau * s
                            for k in range(lj):
                                # A[k,j] += s*A[k,i] # out of band update (DEBUGGING)
                                A[f,k,i] -= g*A[f,k,i]
                            for k in range(ri, rj):
                                # A[f,i,k] -= s*A[f,j,k] # out of band update (DEBUGGING)
                                A[f,j,k] -= g*A[f,j,k]
                    elif ri >= n: # lowest corner of matrix (very large i,j)
                        # assert rj > n # if ri > n also rj > n since ri > rj
                        # ri = rj = n # debugging only
                        for f in range(m):
                            for k in range(lj, i):
                                g = A[f,k,i]
                                h = A[f,k,j]
                                A[f,k,i] -= s*(h + tau*g)
                                A[f,k,j] += s*(g - tau*h)
                            for k in range(j+1, n):
                                g = A[f,i,k]
                                h = A[f,j,k]
                                A[f,i,k] -= s*(h + tau*g)
                                A[f,j,k] += s*(g - tau*h)
                            g = tau * s
                            for k in range(li, lj):
                                # A[f,k,j] += s*A[f,k,i] # out of band update (DEBUGGING)
                                A[f,k,i] -= g*A[f,k,i]
                    elif rj > n: # lower corner of matrix (large i,j)
                        # assert ri <= n # only rj > n
                        # rj = n # debugging only
                        for f in range(m):
                            l = j + 1
                            for k in range(lj, i):
                                g = A[f,k,i]
                                h = A[f,k,j]
                                A[f,k,i] -= s*(h + tau*g)
                                A[f,k,j] += s*(g - tau*h)
                                g = A[f,i,l]
                                h = A[f,j,l]
                                A[f,i,l] -= s*(h + tau*g)
                                A[f,j,l] += s*(g - tau*h)
                                l += 1
                            g = tau * s
                            for k in range(li, lj):
                                # A[f,k,j] += s*A[f,k,i] # out of band update (DEBUGGING)
                                A[f,k,i] -= g*A[f,k,i]
                            for k in range(ri, n):
                                # A[f,i,k] -= s*A[f,j,k] # out of band update (DEBUGGING)
                                A[f,j,k] -= g*A[f,j,k]
                    else: # middle part of matrix 
                        # i < j => li < lj
                        # assert i-lj == ri-(j+1)
                        # assert lj-li == rj-ri
                        for f in range(m):
                            l = j + 1
                            for k in range(lj, i):
                                g = A[f,k,i]
                                h = A[f,k,j]
                                A[f,k,i] -= s*(h + tau*g)
                                A[f,k,j] += s*(g - tau*h)
                                g = A[f,i,l]
                                h = A[f,j,l]
                                A[f,i,l] -= s*(h + tau*g)
                                A[f,j,l] += s*(g - tau*h)
                                l += 1
                            g = tau * s
                            l = ri
                            for k in range(li, lj):
                                # A[f,k,j] += s*A[f,k,i] # out of band update (DEBUGGING)
                                A[f,k,i] -= g*A[f,k,i]
                                # A[f,i,l] -= s*A[f,j,l] # out of band update (DEBUGGING)
                                A[f,j,l] -= g*A[f,j,l]
                                l += 1
                    # now R has to be updated
                    bwR = cRotateR(n, <double*>R.data, bwR, i, j, c, s, tau)
            # stabilize diagonal elements (see Numerical Recipes in C p.467f)
            for f in range(m):
                for i in range(n):
                    b[f,i] += z[f,i]
                    d[f,i] = b[f,i]
                    z[f,i] = 0.0
            if converged == 1: 
                break
        self.d = d # store eigenvalues

        # update the bandwidth of R
        if bwR >= bwMax:
            bwR = cGetBandwidth(R)
        self.bwR = bwR

        # make largest element of each eigenvector positive
        if signConvention: cSetSignsOfEigenvectors(R, bwR) 

        # renormalize vectors of R
        for i in range(n):
            li = i - bwR
            if li < 0: li = 0
            ri = i + bwR + 1
            if ri > n: ri = n
            g = 1./cblas_dnrm2(ri-li, <double*>PyArray_GETPTR2(R, li, i), n) 
            if fabs(1.0 - g) > ZERO:
                # R[li:ri,i] *= g
                cblas_dscal(ri-li, g, <double*>PyArray_GETPTR2(R, li, i), n)

        cdef double* Af = <double*>A.data
        cdef double* df = <double*>d.data
        i = n*n
        if iter >= maxIter and converged == 0:
            if self.A0available != NOT_AVAILABLE or self.finallyDo != TRAFO:
                for f in range(m):
                    cblas_dswap(n, df, 1, Af, n+1)
                    df += n ; Af += i
                raise ValueError('No convergence after %i steps! ' %iter + 
                                    '(Restart possible)')
            else:
                raise ValueError('No convergence after %i steps! ' %iter + 
                                    '(Restart impossible)')
        if printInfo:
            print("Needed %i iterations." %iter)

        # make largest element of each eigenvector positive
        if signConvention: cSetSignsOfEigenvectors(R, bwR) 

        # either return A directly or calculate R.T*A*R
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] C = N.empty( 
                                                    (n,n), dtype = N.float)
        cdef double *R0, *C0
        Af = <double*>A.data
        df = <double*>d.data
        i = n*n
        if self.finallyDo == TRAFO:
            # Now calculate the accurate matrix resulting from the rotations
            R0 = <double*>R.data
            C0 = <double*>C.data
            for f in range(m):
                # restore A0 from lower triangle of A
                if self.A0available == LOWER_TRIANGLE:
                    cSymmetrizeLowerToUpper(n, Af, bwA)
                # restore A0 from upper triangle of A
                elif self.A0available == UPPER_TRIANGLE:
                    cSymmetrizeUpperToLower(n, Af, bwA)
                else: raise AssertionError('Something went terribly wrong!')
                cRtAR(n, Af, R0, bwR, C0)
                Af += i
            # finally set A0available to NOT_AVAILABLE, 
            # since we cannot do a final transformation again, since we lost A0.
            self.A0available = NOT_AVAILABLE
        # No final similarity transformation, just fill A 
        elif self.finallyDo == FILL:
            for f in range(m):
                cblas_dswap(n, df, 1, Af, n+1)
                # build full A from upper triangle and 'd'
                if self.A0available == UPPER_TRIANGLE:
                    cSymmetrizeLowerToUpper(n, Af, bwA)
                else:
                    cSymmetrizeUpperToLower(n, Af, bwA)
                df += n ; Af += i
            self.A0available = NOT_AVAILABLE
        # only fill in diagonal and store diagonal of A0 in d
        elif self.finallyDo == FILL_DIAGONAL:
            for f in range(m):
                cblas_dswap(n, df, 1, Af, n+1)
                df += n ; Af += i
        else: raise AssertionError('Something went terribly wrong!')

        if sort: self._sort() # sort eigenvectors and permute A
        return (A, R) # return eigenvectors as columns

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void _sort(self, sortA = True, bwRupdate = True):
        """
        Sort A according to diagonal elements of first matrix in A, i.e. A[0].
        """
        cdef unsigned int n = self.n
        cdef unsigned int m = self.m
        cdef N.ndarray[N.float_t, ndim = 3] A = self.A
        cdef N.ndarray[N.float_t, ndim = 2] d = self.d
        cdef N.ndarray[N.float_t, ndim = 2] R = self.R
        cdef N.ndarray[SortType_float_int, ndim = 1] sortMe = N.empty(n, 
                           dtype = N.dtype( [('a', N.float), ('i', N.int)], 
                           align = True))
        sortMe['a'] = N.diag(A[0])
        # sortMe['a'] = d.copy() # we have diag(A0) in self.d now
        sortMe['i'] = N.arange(n, dtype = N.int)
        sortMe.sort(order = ['a',])
        cdef N.ndarray[N.int_t, ndim = 1] permutation = sortMe['i']
        self.permutation = permutation
        N.take(R, permutation, axis = 1, out = R)
        cdef int f, nn
        cdef double *C0, *Af
        cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] C 
        if self.A0available == NOT_AVAILABLE:
            for f in range(m):
                # we have diag(A0) in 'd' now so don't permute it
                # N.take(d[f], permutation, axis = 0, out = d[f]) 
                N.take(A[f], permutation, axis = 0, out = A[f]) 
                N.take(A[f], permutation, axis = 1, out = A[f])
        elif self.A0available == LOWER_TRIANGLE:
            C = N.empty((n, n), dtype = N.float)
            Af = <double*>A.data
            nn = n*n
            for f in range(m):
                cPermute(Af, permutation, C)
                Af += nn
        elif self.A0available == UPPER_TRIANGLE:
            raise AssertionError("Not implemented!")
        if bwRupdate: self.bwR = cGetBandwidth(R)


