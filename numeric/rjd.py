import user
from thctk.numeric import *
from math import sqrt, atan2, cos, sin
LA = importLinearAlgebra()

def rjd(A, threshold = 1.0e-8):
#***************************************
# joint diagonalization (possibly
# approximate) of REAL matrices.
#***************************************
# This function minimizes a joint diagonality criterion
# through n matrices of size m by m.
#
# Input :
# * the  n by nm matrix A is the concatenation of m matrices
#   with size n by n. We denote A = [ A1 A2 .... An ]
# * threshold is an optional small number (typically = 1.0e-8 see below).
#
# Output :
# * V is a n by n orthogonal matrix.
# * qDs is the concatenation of (quasi)diagonal n by n matrices:
#   qDs = [ D1 D2 ... Dn ] where A1 = V*D1*V' ,..., An =V*Dn*V'.
#
# The algorithm finds an orthogonal matrix V
# such that the matrices D1,...,Dn  are as diagonal as possible,
# providing a kind of `average eigen-structure' shared
# by the matrices A1 ,..., An.
# If the matrices A1,...,An do have an exact common eigen-structure
# ie a common othonormal set eigenvectors, then the algorithm finds it.
# The eigenvectors THEN are the column vectors of V
# and D1, ...,Dn are diagonal matrices.
# 
# The algorithm implements a properly extended Jacobi algorithm.
# The algorithm stops when all the Givens rotations in a sweep
# have sines smaller than 'threshold'.
# In many applications, the notion of approximate joint diagonalization
# is ad hoc and very small values of threshold do not make sense
# because the diagonality criterion itself is ad hoc.
# Hence, it is often not necessary to push the accuracy of
# the rotation matrix V to the machine precision.
# It is defaulted here to the square root of the machine precision.
#
# Author : Jean-Francois Cardoso. cardoso@sig.enst.fr
# This software is for non commercial use only.
# It is freeware but not in the public domain.
# A version for the complex case is available
# upon request at cardoso@sig.enst.fr
#-----------------------------------------------------
# Two References:
#
# The algorithm is explained in:
#
#@article{SC-siam,
#   HTML =	"ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",
#   author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
#   journal = "{SIAM} J. Mat. Anal. Appl.",
#   title = "Jacobi angles for simultaneous diagonalization",
#   pages = "161--164",
#   volume = "17",
#   number = "1",
#   month = jan,
#   year = {1995}}
#
#  The perturbation analysis is described in
#
#@techreport{PertDJ,
#   author = "{J.F. Cardoso}",
#   HTML =	"ftp://sig.enst.fr/pub/jfc/Papers/joint_diag_pert_an.ps",
#   institution = "T\'{e}l\'{e}com {P}aris",
#   title = "Perturbation of joint diagonalizers. Ref\# 94D027",
#   year = "1994" }
#
#
#
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

if __name__ == '__main__':
    id = N.identity(3, nxFloat)
    a = N.array((( 0.5, 1.0, 2.0),
                 ( 0.7, 3.0, 2.3),
                 ( 0.5, 1.0,-2.0)))
    a = a + N.transpose(a)
    ev = LA.eigenvectors
    E, V = ev(a)
    Vt = N.transpose(V)
    b = id.copy()
    b[1,1] = 2; b[2,2] = 3
    b = N.dot(Vt, N.dot(b, V))
    A = N.array((a, b))
    U, Ad = rjd(A.copy())
