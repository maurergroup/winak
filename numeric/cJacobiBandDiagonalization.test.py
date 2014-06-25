try: from user import *
except: pass
try: from plot import * 
except: pass
import unittest
from random import Random
from time import time
from thctk.numeric import *
from thctk.numeric.cJacobiBandDiagonalization import JacobiBandDiagonalization
# from thctk.numeric.cJacobiBandDiagonalization import SimultaneousJacobiBandDiagonalization, symmetrizeLowerToUpper, symmetrizeUpperToLower, RtAR
from thctk.numeric.cJacobiBandDiagonalization import *
from pdb import set_trace


"""
Do some tests for cJacobiBandDiagonalization (the Cython version), which 
are basically the same as for the python version. 
"""

def normalize(M):
    """ normalize eigenvectors so, that largest element is positive """
    for i in range(len(M)):
        m = M[:,i].argmax()
        n = M[:,i].argmin()
        if N.abs(M[n,i]) > N.abs(M[m,i]):
            M[:,i] = -M[:,i]

def symmetrize(M):
    """ symmetrize matrix """
    for i in range(len(M)):
        for j in range(i):
            M[i,j] = M[j,i]

def givensRotation(A, i, j, theta):
    s = N.sin(theta)
    c = N.cos(theta)
    R = N.identity(len(A))
    R[i,i] = R[j,j] =  c
    R[i,j] = s
    R[j,i] = -s
    A = N.dot(A, R)
    return A

def jacobiRotation(A, i, j, theta):
    s = N.sin(theta)
    c = N.cos(theta)
    R = N.identity(len(A))
    R[i,i] = R[j,j] =  c
    R[i,j] = s
    R[j,i] = -s
    A = N.dot(A, R)
    A = N.dot(R.T, A)
    return A

class Test(unittest.TestCase):

    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.rand = Random()

    def setUp(self):
        A = N.array([
        [1.e-0, 5.e-1, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0 ],
        [  0. , 2.e-0, 1.e-1, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0 ],
        [  0. ,   0. , 2.e-0, 0.e-2, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0 ],
        [  0. ,   0. ,   0. , 3.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0 ],
        [  0. ,   0. ,   0. ,   0. , 1.e-0, 1.e-2, 0.e-0, 0.e-0, 0.e-0, 0.e-0 ],
        [  0. ,   0. ,   0. ,   0. ,   0. , 2.e-0, 0.e-0, 0.e-0, 0.e-0, 0.e-0 ],
        [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,-1.e-0, 0.e-0, 0.e-0, 0.e-0 ],
        [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. , 5.e-0, 1.e-1, 0.e-0 ],
        [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. , 4.e-0, 8.e-1 ],
        [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. , 3.e-0 ],
        ])
        symmetrize(A)
        self.A = A

    def tearDown(self):
        pass

    def test_symmetrize1(self):
        n = 5
        A = N.array([range(n)]*n, N.float)
        B = A.copy()
        symmetrize(B.T)
        symmetrizeLowerToUpper(A)
        self.assert_((A == B).all())

    def test_symmetrize2(self):
        n = 5
        A = N.array([range(n)]*n, N.float)
        B = A.copy()
        symmetrize(B)
        symmetrizeUpperToLower(A)
        self.assert_((A == B).all())

    def test_symmetrize1_band(self):
        n = 5
        A = N.array([range(n)]*n, N.float)
        B = A.copy()
        bw = N.random.randint(1, n)
        for i in range(0, n):
            j = i + bw + 1
            if j > n: j = n
            B[i, i+1:j] = B.T[i, i+1:j]
        symmetrizeLowerToUpper(A, bwA = bw)
        self.assert_((A == B).all())

    def test_symmetrize2_band(self):
        n = 5
        A = N.array([range(n)]*n, N.float)
        B = A.copy()
        bw = N.random.randint(1, n)
        for i in range(n):
            j = i + bw + 1
            if j > n: j = n
            B.T[i, i+1:j] = B[i, i+1:j]
        symmetrizeUpperToLower(A, bwA = bw)
        self.assert_((A == B).all())

    def test_RtAR(self):
        A = self.A
        symmetrize(A)
        n = len(A)
        R = N.array([range(n)]*n, N.float)
        B = A.copy()
        C = N.dot(R.T, N.dot(B, R))
        RtAR(A, R)
        self.assert_(N.allclose(A, C, 1e-15, 1e-15))

    def test_RtAR2(self):
        n = N.random.randint(10, 20)
        A = N.array([N.random.randint(-10**2, 10**2) for i in range(n*n)], N.float)
        A = N.ascontiguousarray(A.reshape((n, n)))
        symmetrize(A)
        n = len(A)
        R = N.array([N.random.randint(-10**2, 10**2) for i in range(n*n)], N.float)
        R = N.ascontiguousarray(R.reshape((n, n)))
        B = A.copy()
        C = N.dot(R.T, N.dot(B, R))
        RtAR(A, R)
        self.assert_(N.allclose(A, C, 1e-15, 1e-15))

    def test_RtAR_band(self):
        n = N.random.randint(500, 550)
        A0 = N.array([N.random.randint(-10**2, 10**2) for i in range(n*n)], N.float)
        A0 = N.ascontiguousarray(A0.reshape((n, n)))
        symmetrizeLowerToUpper(A0)
        R = N.array([N.random.randint(-10**2, 10**2) for i in range(n*n)], N.float)
        R = N.ascontiguousarray(R.reshape((n, n)))
        bwR = N.random.randint(1, n//3)
        for i in range(n-1): R[i, i+1:n].fill(0.)
        symmetrizeLowerToUpper(R, bwA = bwR)
        symmetrizeUpperToLower(R)
        B = A0.copy()
        t = time()
        C = N.dot(R.T, N.dot(B, R))
        print
        print "Full numpy R.T*A*R timing: ", time() - t
        A = A0.copy()
        t = time()
        RtAR(A, R)
        print "Full cblas R.T*A*R timing: ", time() - t
        self.assert_(N.allclose(A, C, 1e-15, 1e-15))
        A = A0.copy()
        t = time()
        RtAR(A, R, bwR)
        print "Band cblas R.T*A*R timing: ", time() - t
        self.assert_(N.allclose(A, C, 1e-15, 1e-15))

    def test_single1(self):
        A = N.diag(N.arange(6, dtype=N.float))
        A[0,2] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = False, finallyDo = 'FILL_DIAGONAL')
        self.assert_( JD.bwR == 2)
        # check that A0 is correct
        self.assert_( N.allclose(JD.d, N.diag(A), 1e-15, 1e-15))
        symmetrizeLowerToUpper(B) # should be A0 (appart from diagonal)
        self.assert_( N.allclose(B - N.diag(N.diag(B)), 
                                 A - N.diag(N.diag(A)), 1e-15, 1e-15))

        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_single2(self):
        A = N.diag(N.arange(6,dtype=N.float))
        A[1,3] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = False, finallyDo = 'FILL_DIAGONAL')
        self.assert_( JD.bwR == 2)
        # check that A0 is correct
        self.assert_( N.allclose(JD.d, N.diag(A), 1e-15, 1e-15))
        symmetrizeLowerToUpper(B) # should be A0 (appart from diagonal)
        self.assert_( N.allclose(B - N.diag(N.diag(B)), 
                                 A - N.diag(N.diag(A)), 1e-15, 1e-15))
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_single3(self):
        A = N.diag(N.arange(9,dtype=N.float))
        A[2,4] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = False, finallyDo = 'FILL_DIAGONAL')
        self.assert_( JD.bwR == 2)
        # check that A0 is correct
        self.assert_( N.allclose(JD.d, N.diag(A), 1e-15, 1e-15))
        symmetrizeLowerToUpper(B) # should be A0 (appart from diagonal)
        self.assert_( N.allclose(B - N.diag(N.diag(B)), 
                                 A - N.diag(N.diag(A)), 1e-15, 1e-15))
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B ,R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_single4(self):
        A = N.diag(N.arange(6,dtype=N.float))
        A[2,3] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = False, finallyDo = 'FILL_DIAGONAL')
        self.assert_( JD.bwR == 1)
        # check that A0 is correct
        self.assert_( N.allclose(JD.d, N.diag(A), 1e-15, 1e-15))
        symmetrizeLowerToUpper(B) # should be A0 (appart from diagonal)
        self.assert_( N.allclose(B - N.diag(N.diag(B)), 
                                 A - N.diag(N.diag(A)), 1e-15, 1e-15))
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B ,R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_single5(self):
        A = N.diag(N.arange(6,dtype=N.float))
        A[4,5] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = False, finallyDo = 'FILL_DIAGONAL')
        self.assert_( JD.bwR == 1)
        # check that A0 is correct
        self.assert_( N.allclose(JD.d, N.diag(A), 1e-15, 1e-15))
        symmetrizeLowerToUpper(B) # should be A0 (appart from diagonal)
        self.assert_( N.allclose(B - N.diag(N.diag(B)), 
                                 A - N.diag(N.diag(A)), 1e-15, 1e-15))
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B ,R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_A2(self):
        A = self.A
        (E, V) = N.linalg.eigh(A)
        normalize(V)

        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
        (B ,R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = False, finallyDo = 'FILL_DIAGONAL')
        # check that A0 is correct
        self.assert_( N.allclose(JD.d, N.diag(A), 1e-15, 1e-15))
        symmetrizeLowerToUpper(B) # should be A0 (appart from diagonal)
        self.assert_( N.allclose(B - N.diag(N.diag(B)), 
                                 A - N.diag(N.diag(A)), 1e-15, 1e-15))
        self.assert_( JD.bwR == 2)

        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
        (B ,R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_A3(self):
        A = self.A
        A[1,3] = 1e-2
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)

        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 3)
        (B ,R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = False, finallyDo = 'FILL_DIAGONAL')
        # check that A0 is correct
        self.assert_( N.allclose(JD.d, N.diag(A), 1e-15, 1e-15))
        symmetrizeLowerToUpper(B) # should be A0 (appart from diagonal)
        self.assert_( N.allclose(B - N.diag(N.diag(B)), 
                                 A - N.diag(N.diag(A)), 1e-15, 1e-15))

        self.assert_( JD.bwR == 3)

        JD = JacobiBandDiagonalization(A, bandwidth = 3)
        (B ,R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_checkR0(self):
        A = self.A
        I = N.identity(len(A))
        for k,l in [(1,0), (1,3), (2,3), (4,5), (2,4), (8,9), (6,8)]:
            (i, j) = (k, l)
            A = self.A.copy()
            for q in range(len(A)):
                A[q,q] += 10

            print i, j
            A[j,i] = A[i,j] = 5.
            JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
            try: 
                (B ,R)=JD(threshold=5.,maxIter = 1,printInfo=True,sort=False)
            except ValueError:
                pass
            R = JD.R
            B = JD.A
            # for q in range(len(B)):
            #     B[q,q] = JD.d[q]
            symmetrize(B)

            d1 = A[i,i] - A[j,j]
            o1 = A[i,j] + A[j,i]
            d2 = d1*d1 - o1*o1
            o2 = 2*d1*o1
            theta = 0.5*N.arctan2(o2, d2 + N.sqrt(d2*d2 + o2*o2) )
            c = N.cos(theta)
            s = N.sin(theta)

            h = A[j,j] - A[i,i]
            theta = 0.5 * h / A[i,j]
            t = 1.0 / (N.abs(theta) + N.sqrt(1.0 + theta*theta))
            if theta < 0.0:
                t = -t
            c1 = 1.0/N.sqrt(1+t*t)
            s1 = t*c
            self.assert_( N.abs(c1 - c) < 1e-15)
            self.assert_( N.abs(-s1 - s) < 1e-15)

            r = N.identity(len(A))
            v1 = N.multiply(r[:,i],  c)
            v2 = N.multiply(r[:,i], -s)
            v3 = N.multiply(r[:,j],  s)
            N.add(v1, v3, r[:,i])
            v3 = N.multiply(r[:,j],  c)
            N.add(v2, v3, r[:,j])

            C = N.dot(r.T, N.dot(A, r))
            C[i,j] = C[j,i] = 0
            self.assert_( N.allclose(
                C[i,i-2 > 0 and i-2 or 0:i+3], 
                B[i,i-2 > 0 and i-2 or 0:i+3], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[j,j-2 > 0 and j-2 or 0:j+3], 
                B[j,j-2 > 0 and j-2 or 0:j+3], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[i-2 > 0 and i-2 or 0:i+3,i], 
                B[i-2 > 0 and i-2 or 0:i+3, i], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[j-2 > 0 and j-2 or 0:j+3,j], 
                B[j-2 > 0 and j-2 or 0:j+3,j], 1e-15, 1e-15) )

            self.assert_( N.allclose(R, r, 1e-15, 1e-15) )
            self.assert_( N.allclose(N.dot(R.T, R), I, 1e-15, 1e-15) )

    def test_checkR1(self):
        A = self.A
        I = N.identity(len(A))
        for k,l in [(1,0), (1,3), (2,3), (4,5), (2,4), (8,9), (6,8)]:
            A = self.A.copy()
            for i in range(len(A)):
                A[i,i] += 10
            for i in range(len(A)):
                for j in range(i-2 > 0 and i-2 or 0, i):
                    A[j,i] = self.rand.random()
            symmetrize(A)
            (i, j) = (k, l)
            A[j,i] = A[i,j] = 5.
            JD = JacobiBandDiagonalization(A.copy(), bandwidth = 2)
            try: 
                (B ,R)=JD(threshold=5.,maxIter = 1,printInfo=True,sort=False)
            except ValueError:
                pass
            R = JD.R
            B = JD.A
            # for q in range(len(B)):
            #     B[q,q] = JD.d[q]
            symmetrize(B)

            d1 = A[i,i] - A[j,j]
            o1 = A[i,j] + A[j,i]
            d2 = d1*d1 - o1*o1
            o2 = 2*d1*o1
            theta = 0.5*N.arctan2(o2, d2 + N.sqrt(d2*d2 + o2*o2) )
            c = N.cos(theta)
            s = N.sin(theta)

            h = A[j,j] - A[i,i]
            theta = 0.5 * h / A[i,j]
            t = 1.0 / (N.abs(theta) + N.sqrt(1.0 + theta*theta))
            if theta < 0.0:
                t = -t
            c1 = 1.0/N.sqrt(1+t*t)
            s1 = t*c
            self.assert_( N.abs(c1 - c) < 1e-15)
            self.assert_( N.abs(-s1 - s) < 1e-15)

            r = N.identity(len(A))
            v1 = N.multiply(r[:,i],  c)
            v2 = N.multiply(r[:,i], -s)
            v3 = N.multiply(r[:,j],  s)
            N.add(v1, v3, r[:,i])
            v3 = N.multiply(r[:,j],  c)
            N.add(v2, v3, r[:,j])

            C = N.dot(r.T, N.dot(A, r))
            C[i,j] = C[j,i] = 0
            self.assert_( N.allclose(
                C[i,i-2 > 0 and i-2 or 0:i+3], 
                B[i,i-2 > 0 and i-2 or 0:i+3], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[j,j-2 > 0 and j-2 or 0:j+3], 
                B[j,j-2 > 0 and j-2 or 0:j+3], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[i-2 > 0 and i-2 or 0:i+3,i], 
                B[i-2 > 0 and i-2 or 0:i+3, i], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[j-2 > 0 and j-2 or 0:j+3,j], 
                B[j-2 > 0 and j-2 or 0:j+3,j], 1e-15, 1e-15) )

            self.assert_( N.allclose(R, r, 1e-15, 1e-15) )
            self.assert_( N.allclose(N.dot(R.T, R), I, 1e-15, 1e-15) )

    def test_checkR1_sim(self):
        A = self.A
        I = N.identity(len(A))
        for k,l in [(1,0), (1,3), (2,3), (4,5), (2,4), (8,9), (6,8)]:
            A = self.A.copy()
            for i in range(len(A)):
                A[i,i] += 10
            for i in range(len(A)):
                for j in range(i-2 > 0 and i-2 or 0, i):
                    A[j,i] = self.rand.random()*0.2
            symmetrize(A)
            (i, j) = (k, l)
            A[j,i] = A[i,j] = 5
            JD = SimultaneousJacobiBandDiagonalization(N.array([A.copy()]), )
                                            # bandwidth = 2)
            try: 
                (B ,R)=JD(threshold=5,maxIter = 1,printInfo=True,sort=False, 
                            finallyDo = 'fill')
            except ValueError:
                pass
            R = JD.R
            B = JD.A[0]
            # for q in range(len(B)):
            #     B[q,q] = JD.d[0,q]
            symmetrize(B)

            d1 = A[i,i] - A[j,j]
            o1 = A[i,j] + A[j,i]
            d2 = d1*d1 - o1*o1
            o2 = 2*d1*o1
            theta = 0.5*N.arctan2(o2, d2 + N.sqrt(d2*d2 + o2*o2) )
            c = N.cos(theta)
            s = N.sin(theta)

            h = A[j,j] - A[i,i]
            theta = 0.5 * h / A[i,j]
            t = 1.0 / (N.abs(theta) + N.sqrt(1.0 + theta*theta))
            if theta < 0.0:
                t = -t
            c1 = 1.0/N.sqrt(1+t*t)
            s1 = t*c
            self.assert_( N.abs(c1 - c) < 1e-15)
            self.assert_( N.abs(-s1 - s) < 1e-15)

            r = N.identity(len(A))
            v1 = N.multiply(r[:,i],  c)
            v2 = N.multiply(r[:,i], -s)
            v3 = N.multiply(r[:,j],  s)
            N.add(v1, v3, r[:,i])
            v3 = N.multiply(r[:,j],  c)
            N.add(v2, v3, r[:,j])

            C = N.dot(r.T, N.dot(A, r))
            C[i,j] = C[j,i] = 0
            self.assert_( N.allclose(
                C[i,i-2 > 0 and i-2 or 0:i+3], 
                B[i,i-2 > 0 and i-2 or 0:i+3], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[j,j-2 > 0 and j-2 or 0:j+3], 
                B[j,j-2 > 0 and j-2 or 0:j+3], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[i-2 > 0 and i-2 or 0:i+3,i], 
                B[i-2 > 0 and i-2 or 0:i+3, i], 1e-15, 1e-15) )
            self.assert_( N.allclose(
                C[j-2 > 0 and j-2 or 0:j+3,j], 
                B[j-2 > 0 and j-2 or 0:j+3,j], 1e-15, 1e-15) )

            self.assert_( N.allclose(R, r, 1e-15, 1e-15) )
            self.assert_( N.allclose(N.dot(R.T, R), I, 1e-15, 1e-15) )

    def test_sim2(self):
        A = N.diag(N.arange(5, dtype = N.float))
        B = N.diag(N.arange(5, dtype = N.float))*0.1
        A0 = A.copy()
        B0 = B.copy()
        R = N.identity(len(A))
        for k in range(3):
            i = N.random.randint(1, len(A))
            j = N.random.randint(i)
            theta = N.random.random()*N.pi/4.
            A = jacobiRotation(A, i, j, theta)
            B = jacobiRotation(B, i, j, theta)
            R = givensRotation(R, i, j, theta)
        R = R.T
        for i in range(len(R)):
            j = N.abs(R[:,i]).argmax()
            if R[j,i] < 0.:
                R[:,i] *= -1

        D = SimultaneousJacobiBandDiagonalization(N.array([A,B]))
        (H, r) = D(threshold = 1e-15, sort = True)
        self.assert_(N.allclose(H[0], A0, 1e-15, 1e-15))
        self.assert_(N.allclose(H[1], B0, 1e-15, 1e-15))
        try:
            self.assert_(N.allclose(r, R, 1e-15, 1e-15))
        except AssertionError:
            set_trace()

    def test_sim3(self):
        A = N.diag(N.arange(5, dtype = N.float))
        B = N.diag(N.arange(5, dtype = N.float))*0.1
        C = N.diag(N.arange(5, dtype = N.float))*0.01
        A0 = A.copy()
        B0 = B.copy()
        C0 = C.copy()
        R = N.identity(len(A))
        for k in range(3):
            i = N.random.randint(1, len(A))
            j = N.random.randint(i)
            theta = N.random.random()*N.pi/4.
            A = jacobiRotation(A, i, j, theta)
            B = jacobiRotation(B, i, j, theta)
            C = jacobiRotation(C, i, j, theta)
            R = givensRotation(R, i, j, theta)
        R = R.T
        for i in range(len(R)):
            j = N.abs(R[:,i]).argmax()
            if R[j,i] < 0.:
                R[:,i] *= -1

        D = SimultaneousJacobiBandDiagonalization(N.array([A, B, C]))
        (H, r) = D(threshold = 1e-15, sort = True)
        self.assert_(N.allclose(H[0], A0, 1e-15, 1e-15))
        self.assert_(N.allclose(H[1], B0, 1e-15, 1e-15))
        self.assert_(N.allclose(H[2], C0, 1e-15, 1e-15))
        try:
            self.assert_(N.allclose(r, R, 1e-15, 1e-15))
        except AssertionError:
            set_trace()

    def test_simBand(self):
        A = N.diag(N.arange(100, dtype = N.float))
        B = N.diag(N.arange(100, dtype = N.float))*0.1
        A0 = A.copy()
        B0 = B.copy()
        R = N.identity(len(A))
        for k in range(10):
            i = N.random.randint(5, len(A)-5)
            j = N.random.randint(i-4, i)
            theta = N.random.random()*N.pi/4.
            A = jacobiRotation(A, i, j, theta)
            B = jacobiRotation(B, i, j, theta)
            R = givensRotation(R, i, j, theta)
        symmetrizeLowerToUpper(A)
        symmetrizeLowerToUpper(B)
        R = R.T
        for i in range(len(R)):
            j = N.abs(R[:,i]).argmax()
            if R[j,i] < 0.:
                R[:,i] *= -1

        D = SimultaneousJacobiBandDiagonalization(N.array([A,B]), 
                                                            bandwidth = 20)
        (H, r) = D(threshold = 1e-16, printInfo=True, sort = False, 
                    finallyDo = 'FILL_DIAGONAL')
        self.assert_(N.allclose(A, 
                N.tril(H[0], k = -1) + N.tril(H[0], k = -1).T + N.diag(D.d[0]),
                1e-16, 1e-16))
        self.assert_(N.allclose(B, 
                N.tril(H[1], k = -1) + N.tril(H[1], k = -1).T + N.diag(D.d[1]),
                1e-16, 1e-16))

        p = N.argsort(N.diag(H[0]))
        r = N.take(r, p, axis = 1)
        RtAR(A, r)
        RtAR(B, r)
        print N.max(N.abs(A)-N.abs(A0))
        print N.max(N.abs(B)-N.abs(B0))
        self.assert_(N.allclose(A, A0, 1e-15, 1e-13))
        self.assert_(N.allclose(B, B0, 1e-15, 1e-13))
        self.assert_(N.allclose(r, R, 1e-15, 1e-13))

        # diagonalize further
        (H, r) = D(threshold=1e-16, printInfo=True, sort=True, 
                    finallyDo = 'TRAFO')
        self.assert_(N.allclose(H[0], A0, 1e-15, 1e-14))
        self.assert_(N.allclose(H[1], B0, 1e-15, 1e-14))
        self.assert_(N.allclose(r, R, 1e-15, 1e-14))

    def test_overflow(self):
        A = N.diag(N.arange(3, dtype = N.float))
        A[0,0] = 1.0e17 
        A[0,1] = 1.0e-1
        A[1,1] = -2.0e17 
        symmetrize(A)
        JD = JacobiBandDiagonalization(A.copy(), bandwidth = 1)
        (B ,R) = JD(threshold=1e-16, maxIter=2, printInfo=True, sort=True)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        normalize(R)
        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_largeM(self):
        n = 500
        n = 100
        bw = n // 50
        A = N.diag(N.arange(n, dtype=N.float)/n)
        def rotate(A, theta, i, j):
            c = N.cos(theta)
            s = N.sin(theta)
            a1 = N.multiply(A[:,i],  c)
            a2 = N.multiply(A[:,i], -s)
            a3 = N.multiply(A[:,j],  s)
            N.add(a1, a3, A[:,i])
            a3 = N.multiply(A[:,j],  c, a3)
            N.add(a2, a3, A[:,j])
            a1 = N.multiply(A[i,:],  c, a1)
            a2 = N.multiply(A[i,:], -s, a2)
            a3 = N.multiply(A[j,:],  s, a3)
            N.add(a1, a3, A[i,:])
            a3 = N.multiply(A[j,:],  c, a3)
            N.add(a2, a3, A[j,:])
            # a1 = N.multiply(R[:,i],  c, a1)
            # a2 = N.multiply(R[:,i], -s, a2)
            # a3 = N.multiply(R[:,j],  s, a3)
            # N.add(a1, a3, R[:,i])
            # a3 = N.multiply(R[:,j],  c, a3)
            # N.add(a2, a3, R[:,j])
        for k in range(n):
            i = N.random.randint(1, n)
            if i < bw: l = 0
            else: l = i-bw
            j = N.random.randint(l, i)
            theta = (N.random.random()-N.pi)*2*N.pi # theta within [-pi,pi]
            rotate(A, theta, i, j)
        symmetrizeLowerToUpper(A)

        A0 = A.copy()
        JD = JacobiBandDiagonalization(A, bandwidth = n // 3)
        print "\nSolving large A ..."
        (B ,R) = JD(threshold=1e-16, maxIter=10, printInfo=True, sort=False, 
                    finallyDo = 'FILL_DIAGONAL')
        p = N.argsort(N.diag(B))
        R = N.take(R, p, axis = 1)
        B = A0.copy()
        RtAR(B, R)
        try:
            self.assert_( N.allclose(B, N.diag(N.arange(n, dtype=N.float)/n), 1e-14, 1e-14))
        except AssertionError:
            D = B-N.diag(N.arange(n, dtype=N.float)/n)
            print D.min()
            print D.max()
            print N.unravel_index(D.argmin(), B.shape)
            print N.unravel_index(D.argmax(), B.shape)
            raise
        self.assert_( N.allclose(N.dot(R.T, R), N.identity(n), 1e-15, 1e-14) )

        # diagonalize further
        (B, R) = JD(threshold=1e-16, maxIter=10, printInfo=True, sort=True, 
                    finallyDo = 'TRAFO')
        try:
            self.assert_( N.allclose(B, N.diag(N.arange(n, dtype=N.float)/n), 1e-14, 1e-14))
        except AssertionError:
            D = B-N.diag(N.arange(n, dtype=N.float)/n)
            print D.min()
            print D.max()
            print N.unravel_index(D.argmin(), B.shape)
            print N.unravel_index(D.argmax(), B.shape)
            raise
        self.assert_( N.allclose(N.dot(R.T, R), N.identity(n), 1e-15, 1e-14) )
        print "... done."


    def test_fullLargeM(self):
        n = 100
        bw = n // 10
        A = N.diag(N.arange(n, dtype=N.float)/n)
        def rotate(A, theta, i, j):
            c = N.cos(theta)
            s = N.sin(theta)
            a1 = N.multiply(A[:,i],  c)
            a2 = N.multiply(A[:,i], -s)
            a3 = N.multiply(A[:,j],  s)
            N.add(a1, a3, A[:,i])
            a3 = N.multiply(A[:,j],  c, a3)
            N.add(a2, a3, A[:,j])
            a1 = N.multiply(A[i,:],  c, a1)
            a2 = N.multiply(A[i,:], -s, a2)
            a3 = N.multiply(A[j,:],  s, a3)
            N.add(a1, a3, A[i,:])
            a3 = N.multiply(A[j,:],  c, a3)
            N.add(a2, a3, A[j,:])
            # a1 = N.multiply(R[:,i],  c, a1)
            # a2 = N.multiply(R[:,i], -s, a2)
            # a3 = N.multiply(R[:,j],  s, a3)
            # N.add(a1, a3, R[:,i])
            # a3 = N.multiply(R[:,j],  c, a3)
            # N.add(a2, a3, R[:,j])
        for k in range(n):
            i = N.random.randint(1, n)
            if i < bw: l = 0
            else: l = i-bw
            j = N.random.randint(l, i)
            theta = (N.random.random()-N.pi)*2*N.pi # theta within [-pi,pi]
            rotate(A, theta, i, j)
        A0 = A.copy()
        JD = JacobiBandDiagonalization(A, bandwidth = 0)
        print "\nSolving full large A ..."
        (B ,R) = JD(threshold=1e-16, maxIter=10, printInfo=True, sort=True, 
                    finallyDo = 'FILL_DIAGONAL')
        B = A0.copy()
        RtAR(B, R)
        self.assert_( N.allclose(B, N.diag(N.arange(n, dtype=N.float)/n), 1e-14, 1e-14))
        self.assert_( N.allclose(N.dot(R.T, R), N.identity(n), 1e-14, 1e-14) )

        # diagonalize further
        (B, R) = JD(threshold=1e-16, maxIter=10, printInfo=True, sort=True, 
                    finallyDo = 'TRAFO')
        self.assert_( N.allclose(B, N.diag(N.arange(n, dtype=N.float)/n), 1e-14, 1e-14))
        self.assert_( N.allclose(N.dot(R.T, R), N.identity(n), 1e-14, 1e-14) )

        print "... done."

    def test_RayleighQuotients(self):
        n = 15
        bwR = 3
        R = N.ones((n,n), dtype = N.float)
        R = N.tril(R)
        symmetrizeLowerToUpper(R, bwA = bwR)
        symmetrizeUpperToLower(R)
        A = N.array([range(n)]*n, dtype = N.float)
        r = N.diag(N.dot(N.dot(R.T, A), R))/N.diag(N.dot(R.T, R))

        print "1"
        rr = rayleighQuotients(A, R)
        self.assert_(N.allclose(rr, r, 1e-15, 1e-15))

        print "2"
        rr = rayleighQuotients(A, R, bwR = bwR)
        self.assert_(N.allclose(rr, r, 1e-15, 1e-15))

        R /= N.sqrt(N.diag(N.dot(R.T, R)))[N.newaxis,:]

        print "3"
        rr = rayleighQuotients(A, R, normR = False)
        self.assert_(N.allclose(rr, r, 1e-15, 1e-15))

        print "4"
        rr = rayleighQuotients(A, R, bwR = bwR, normR = False)
        self.assert_(N.allclose(rr, r, 1e-15, 1e-15))

            
unittest.main()

