import user
from thctk.numeric import *
from math import sqrt, atan2, cos, sin
LA = importLinearAlgebra()

def RJD(A, threshold = 1.0e-8):
    cdef int k, m, n, p, q
    A = N.array(A)
    if nxTypecode(A) != nxFloat:
        raise TypeError('only real matrices are supported')
    k, m, n = A.shape
    if k < 2:
        raise IndexError('less than 2 matrices to diagonalize')
    if m != n:
        raise IndexError('matrices have to be square')
    V = N.identity(m, nxFloat)
    # the following matrices are used to hold temporary data
    a1 = N.zeros((k, m), nxFloat)
    a2 = N.zeros((k, m), nxFloat)
    a3 = N.zeros((k, m), nxFloat)

    encore = True
    while encore:
        encore = False
        for p in range(m):
            for q in range(p+1, m):
                # computation of Givens rotations
                am = N.subtract(A[:,p,p], A[:,q,q], a1[:,0])
                ap = N.add(A[:,p,q], A[:,q,p], a2[:,0])
                ton  = N.dot(am, am) - N.dot(ap, ap)
                toff = 2*N.dot(am, ap)
                theta = 0.5*atan2( toff , ton + sqrt(ton*ton + toff*toff) )
                c = cos(theta)
                s = sin(theta)
                encore = encore or (abs(s) > threshold)
                if abs(s) > threshold:
                    # update of the A and V matrices 
                    # avoid creating temporary matrices
                    a1 = N.multiply(A[:,:,p],  c, a1)
                    a2 = N.multiply(A[:,:,p], -s, a2)
                    a3 = N.multiply(A[:,:,q],  s, a3)
                    N.add(a1, a3, A[:,:,p])
                    a3 = N.multiply(A[:,:,q],  c, a3)
                    N.add(a2, a3, A[:,:,q])
                    a1 = N.multiply(A[:,p,:],  c, a1)
                    a2 = N.multiply(A[:,p,:], -s, a2)
                    a3 = N.multiply(A[:,q,:],  s, a3)
                    N.add(a1, a3, A[:,p,:])
                    a3 = N.multiply(A[:,q,:],  c, a3)
                    N.add(a2, a3, A[:,q,:])
                    v1 = N.multiply(V[:,p],  c, a1[0])
                    v2 = N.multiply(V[:,p], -s, a2[0])
                    v3 = N.multiply(V[:,q],  s, a3[0])
                    N.add(v1, v3, V[:,p])
                    v3 = N.multiply(V[:,q],  c, a3[0])
                    N.add(v2, v3, V[:,q])
    qDs = A
    return V, qDs

