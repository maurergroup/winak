import numpy as N
cimport numpy as N
cimport cython

N.import_array()

cdef extern from "numpy/arrayobject.h":
    void* PyArray_GETPTR2(N.ndarray obj, int i, int j)

cdef extern from "math.h":
    double M_PI
    double fabs(double x)
    double sin(double x)

cdef extern from "cblas.h":
    enum CBLAS_ORDER:     
        CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: 
        CblasNoTrans, CblasTrans, CblasConjTrans
    void cblas_dscal(int n, double alpha, double *x, int incx)

cdef inline double dampingFunction(double x, double x0):
    cdef double t
    t = sin((M_PI/2.*x)/x0)
    t *= t ; t *= t ; t *= t
    return t

@cython.wraparound(False)
@cython.boundscheck(False)
def normalizeIC(q_in, stretch_in, bends_in, bendsInOthers_in, tors_in, oop_in):
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] q = q_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] stretch = stretch_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] bends = bends_in
    # cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] bendsInOthers=bendsInOthers_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] tors = tors_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] oop = oop_in
    cdef int i
    cdef double *qi

    # set stretches to [0, inf]
    for i in range(stretch.shape[0]):
        qi = &(q[stretch[i]])
        if qi[0] < 0.: qi[0] = -qi[0]

    # set bends to [0, pi]
    # for i in range(0, bends.shape[0], 2): # this one is translated badly.
    for i from 1 <= i < bends.shape[0] by 2:
        qi = &(q[bends[i]])
        if   qi[0] > M_PI: qi[0] = 2*M_PI - qi[0]
        elif qi[0] < 0.: qi[0] = -qi[0]

    # set torsions to [-pi, pi]
    for i in range(tors.shape[0]):
        qi = &(q[tors[i]])
        if fabs(qi[0]) > M_PI:
            if (fabs(qi[0] + 2*M_PI) < M_PI): qi[0] += 2*M_PI
            else:                             qi[0] -= 2*M_PI

    # set out-of-plains to [-pi/2, pi/2]
    for i in range(oop.shape[0]):
        qi = &(q[oop[i]])
        if fabs(qi[0]) > M_PI/2.:
            if qi[0] >  M_PI/2: qi[0] =  M_PI - qi[0]
            else:               qi[0] = -M_PI - qi[0]

@cython.wraparound(False)
@cython.boundscheck(False)
def denseDampIC(printWarning, q_in, dq_in,  B_in, double dampingThreshold, 
                    stretch_in, bends_in, bendsInOthers_in, tors_in, oop_in):
    """
    Note, the format of these arrays is described in 'InternalCoordinates.py'.
    """
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] q = q_in
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] dq = dq_in
    cdef N.ndarray[N.float_t, ndim = 2, mode = 'c'] B = B_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] stretch = stretch_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] bends = bends_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] bendsInOthers=bendsInOthers_in
    # cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] tors = tors_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] oop = oop_in
    cdef int i, j, k, l
    cdef double *qi, t

    for i in range(stretch.shape[0]):
        qi = &(q[stretch[i]])
        if qi[0] < dampingThreshold: 
            t = dampingFunction(qi[0], dampingThreshold)
            k = stretch[i]
            dq[k] *= t
            cblas_dscal(B.shape[1], t, <double*>PyArray_GETPTR2(B, k, 0), 1)
            # print "S %i %5.3f %2.1e" %(stretch[i], qi[0], t)

    # for i in range(0, bends.shape[0], 2): # this one is translated badly.
    for i from 1 <= i < bends.shape[0] by 2:
        qi = &(q[bends[i]])
        if qi[0] > M_PI - dampingThreshold:
            t = dampingFunction(M_PI - qi[0], dampingThreshold)
            k = bends[i]
            dq[k] *= t
            cblas_dscal(B.shape[1], t, <double*>PyArray_GETPTR2(B, k, 0), 1)
            printWarning(k, qi[0], 'pi')
            # print "B %i %5.3f %2.1e" %(bends[i], qi[0], t)
            for j in range(bends[i-1], bends[i+1]):
                l = bendsInOthers[j]
                dq[l] *= t 
                cblas_dscal(B.shape[1], t, <double*>PyArray_GETPTR2(B, l, 0), 1)
                # print "C/O %i %5.3f %2.1e" %(l, q[l], t)
        elif qi[0] < dampingThreshold:
            t = dampingFunction(qi[0], dampingThreshold)
            k = bends[i]
            dq[k] *= t
            cblas_dscal(B.shape[1], t, <double*>PyArray_GETPTR2(B, k, 0), 1)
            printWarning(k, qi[0], '0')
            for j in range(bends[i-1], bends[i+1]):
                l = bendsInOthers[j]
                dq[l] *= t
                cblas_dscal(B.shape[1], t, <double*>PyArray_GETPTR2(B, l, 0), 1)
                # print "C/O %i %5.3f %2.1e" %(l, q[l], t)

    for i in range(oop.shape[0]):
        qi = &(q[oop[i]])
        if M_PI/2. - fabs(qi[0]) < dampingThreshold:
            t = dampingFunction(M_PI/2 - fabs(qi[0]), dampingThreshold)
            k = oop[i]
            dq[k] *= t
            cblas_dscal(B.shape[1], t, <double*>PyArray_GETPTR2(B, k, 0), 1)
            printWarning(k, qi[0], 'pi/2')
            # print "O %i %5.3f %2.1e" %(oop[i], qi[0], t)

@cython.wraparound(False)
@cython.boundscheck(False)
def sparseDampIC(printWarning, q_in, dq_in, B, double dampingThreshold, 
                    stretch_in, bends_in, bendsInOthers_in, tors_in, oop_in):
    """
    Note, the format of these arrays is described in 'InternalCoordinates.py'.
    """
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] q = q_in
    cdef N.ndarray[N.float_t, ndim = 1, mode = 'c'] dq = dq_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] stretch = stretch_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] bends = bends_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] bendsInOthers=bendsInOthers_in
    # cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] tors = tors_in
    cdef N.ndarray[N.int_t, ndim = 1, mode = 'c'] oop = oop_in
    cdef int i, j, k, l
    cdef double *qi, t

    for i in range(stretch.shape[0]):
        qi = &(q[stretch[i]])
        if qi[0] < dampingThreshold: 
            t = dampingFunction(qi[0], dampingThreshold)
            k = stretch[i]
            dq[k] *= t
            B.scaleRow(k, t)
            # print "S %i %5.3f %2.1e" %(stretch[i], qi[0], t)

    # for i in range(0, bends.shape[0], 2): # this one is translated badly.
    for i from 1 <= i < bends.shape[0] by 2:
        qi = &(q[bends[i]])
        if qi[0] > M_PI - dampingThreshold:
            t = dampingFunction(M_PI - qi[0], dampingThreshold)
            k = bends[i]
            dq[k] *= t
            B.scaleRow(k, t)
            printWarning(k, qi[0], 'pi')
            # print "B %i %5.3f %2.1e" %(bends[i], qi[0], t)
            for j in range(bends[i-1], bends[i+1]):
                l = bendsInOthers[j]
                dq[l] *= t 
                B.scaleRow(l, t)
                # print "C/O %i %5.3f %2.1e" %(l, q[l], t)
        elif qi[0] < dampingThreshold:
            t = dampingFunction(qi[0], dampingThreshold)
            k = bends[i]
            dq[k] *= t
            B.scaleRow(k, t)
            printWarning(k, qi[0], '0')
            for j in range(bends[i-1], bends[i+1]):
                l = bendsInOthers[j]
                dq[l] *= t
                B.scaleRow(l, t)
                # print "C/O %i %5.3f %2.1e" %(l, q[l], t)

    for i in range(oop.shape[0]):
        qi = &(q[oop[i]])
        if M_PI/2. - fabs(qi[0]) < dampingThreshold:
            t = dampingFunction(M_PI/2 - fabs(qi[0]), dampingThreshold)
            k = oop[i]
            dq[k] *= t
            B.scaleRow(k, t)
            printWarning(k, qi[0], 'pi/2')
            # print "O %i %5.3f %2.1e" %(oop[i], qi[0], t)

################################ Old Python Code ###############################
# 
# # set stretches to [0, inf]
# for (i, b_i) in zip(stretch, N.take(q, stretch)):
#     if b_i < 0.: q[i] = -b_i
# 
# # set bends to [0, pi]
# for (i, b_i) in zip(bend, N.take(q, bend)):
#     if   b_i > N.pi: q[i] = 2*N.pi - b_i
#     elif b_i < 0.:   q[i] = -b_i
# 
# # set out-of-plains to [-pi/2, pi/2]
# for (i, b_i) in zip(oop, N.take(q, oop)):
#     if   b_i >  N.pi/2.: q[i] =  N.pi - b_i
#     elif b_i < -N.pi/2.: q[i] = -N.pi - b_i
# 
# # bring target geometry to standard form
# q = ic.dphi_mod_2pi(q, torsions)
# 
# # stretches
# for (i, b_i) in zip(stretch, N.take(qn, stretch)):
#     if b_i < dampingThreshold: 
#         t = N.sin(N.pi/2.*b_i/dampingThreshold)
#         t *= t ; t *= t ; t *= t
#         dq[i] *= t ; B[i] *= t
#         print "s %i %5.3f %2.1e" %(i, b_i, t)
# 
# # bends
# for (i, b_i) in zip(bend, N.take(qn, bend)):
#     if b_i > N.pi - dampingThreshold:
#         printWarning(i, b_i, 'pi')
#         t = N.sin(N.pi/2.*(N.pi - b_i)/dampingThreshold)
#         t *= t ; t *= t ; t *= t
#         dq[i] *= t ; B[i] *= t
#         print "b %i %5.3f %2.1e" %(i, b_i, t)
#         if i in bendsInOthers:
#             for j in bendsInOthers[i]:
#                 dq[j] *= t ; B[j] *= t
#                 print "t/o %i %5.3f %2.1e" %(j, qn[j], t)
#     elif b_i < 0. + dampingThreshold:   
#         printWarning(i, b_i, '0')
#         t = N.sin(N.pi/2.*b_i/dampingThreshold)
#         t *= t ; t *= t ; t *= t
#         dq[i] *= t ; B[i] *= t
#         print "b %i %5.3f %2.1e" %(i, b_i, t)
#         if i in bendsInOthers:
#             for j in bendsInOthers[i]:
#                 dq[j] *= t ; B[j] *= t
#                 print "t/o %i %5.3f %2.1e" %(j, qn[j], t)
# 
# # out-of-plains
# for (i, b_i) in zip(oop, N.take(qn, oop)):
#     if b_i >  N.pi/2. - dampingThreshold: 
#         printWarning(i, b_i, 'pi/2')
#         t = N.sin(N.pi/2.*(N.pi/2. - b_i)/dampingThreshold)
#         t *= t ; t *= t ; t *= t
#         dq[i] *= t ; B[i] *= t
#         print "o %i %5.3f %2.1e" %(i, b_i, t)
#     elif b_i < -N.pi/2. + dampingThreshold: 
#         printWarning(i, b_i, 'pi/2')
#         t = N.sin(N.pi/2.*(N.pi/2. + b_i)/dampingThreshold)
#         t *= t ; t *= t ; t *= t
#         dq[i] *= t ; B[i] *= t
#         print "o %i %5.3f %2.1e" %(i, b_i, t)
# 
