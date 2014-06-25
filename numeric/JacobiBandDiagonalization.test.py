from user import *
import unittest
from random import Random
from thctk.numeric import *
from thctk.numeric.JacobiBandDiagonalization import JacobiBandDiagonalization

"""
Do some tests for JacobiBandDiagonalization (the Python version).
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

    def test_single1(self):
        A = N.diag(N.arange(6,dtype=N.float))
        A[0,2] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )
        self.assert_( JD.bwR == 2)

    def test_single2(self):
        A = N.diag(N.arange(6,dtype=N.float))
        A[1,3] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )
        self.assert_( JD.bwR == 2)

    def test_single3(self):
        A = N.diag(N.arange(9,dtype=N.float))
        A[2,4] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )
        self.assert_( JD.bwR == 2)

    def test_single4(self):
        A = N.diag(N.arange(6,dtype=N.float))
        A[2,3] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )
        self.assert_( JD.bwR == 2)

    def test_single5(self):
        A = N.diag(N.arange(6,dtype=N.float))
        A[4,5] = 2.
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )
        self.assert_( JD.bwR == 2)


    def test_A2(self):
        A = self.A
        (E, V) = N.linalg.eigh(A)
        normalize(V)

        JD = JacobiBandDiagonalization(A, bandwidth = 2)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )
        self.assert_( JD.bwR == 2)

    def test_A3(self):
        A = self.A
        A[1,3] = 1e-2
        symmetrize(A)
        (E, V) = N.linalg.eigh(A)
        normalize(V)

        JD = JacobiBandDiagonalization(A, bandwidth = 3)
        (B, R) = JD(threshold = 1e-16, maxIter = 20, printInfo = True, sort = True)
        normalize(R)

        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )
        self.assert_( JD.bwR == 3)

    def test_checkR1(self):
        A = self.A
        for i in range(len(A)):
            A[i,i] += 10

        I = N.identity(len(A))
        for k,l in [(1,0), (1,3), (2,3), (4,5), (2,4), (8,9), (6,8)]:
            for i in range(len(A)):
                for j in range(i-2 > 0 and i-2 or 0, i):
                    A[j,i] = self.rand.random()
            symmetrize(A)
            (i, j) = (k, l)
            A[j,i] = A[i,j] = 5.
            JD = JacobiBandDiagonalization(A, bandwidth = 2)
            try: 
                (B, R)=JD(threshold=5.,maxIter = 1,printInfo=True,sort=False)
            except ValueError:
                pass
            R = JD.R
            B = JD.A
            symmetrize(B)

            d1 = A[i,i] - A[j,j]
            o1 = A[i,j] + A[j,i]
            d2 = d1*d1 - o1*o1
            o2 = 2*d1*o1
            theta = 0.5*N.arctan2(o2, d2 + N.sqrt(d2*d2 + o2*o2) )
            c = N.cos(theta)
            s = N.sin(theta)
            r = N.identity(len(A))
            v1 = N.multiply(r[:,i],  c)
            v2 = N.multiply(r[:,i], -s)
            v3 = N.multiply(r[:,j],  s)
            N.add(v1, v3, r[:,i])
            v3 = N.multiply(r[:,j],  c)
            N.add(v2, v3, r[:,j])

            C = N.dot(r.T, N.dot(A, r))
            C[i,j] = 0
            self.assert_( N.allclose(
                C[i,i-2 > 0 and i-2 or 0:i+3], 
                B[i,i-2 > 0 and i-2 or 0:i+3], 1e-14, 1e-14) )
            self.assert_( N.allclose(
                C[j,j-2 > 0 and j-2 or 0:j+3], 
                B[j,j-2 > 0 and j-2 or 0:j+3], 1e-14, 1e-14) )
            self.assert_( N.allclose(
                C[i-2 > 0 and i-2 or 0:i+3,i], 
                B[i-2 > 0 and i-2 or 0:i+3, i], 1e-14, 1e-14) )
            self.assert_( N.allclose(
                C[j-2 > 0 and j-2 or 0:j+3,j], 
                B[j-2 > 0 and j-2 or 0:j+3,j], 1e-14, 1e-14) )

            self.assert_( N.allclose(R, r, 1e-15, 1e-15) )
            self.assert_( N.allclose(N.dot(R.T, R), I, 1e-15, 1e-15) )

    def test_overflow(self):
        A = N.diag(N.arange(100, dtype = N.float))
        A[50,50] = 1.0e17
        A[50,51] = 1.0e-1
        A[51,51] = -2.0e17
        symmetrize(A)
        JD = JacobiBandDiagonalization(A, bandwidth = 33)
        (B, R) = JD(threshold=1e-16, maxIter=10, printInfo=True, sort=True)
        (E, V) = N.linalg.eigh(A)
        normalize(V)
        normalize(R)
        self.assert_( N.allclose(N.diag(B), E, 1e-15, 1e-15) )
        self.assert_( N.allclose(R, V, 1e-15, 1e-15) )

    def test_largeM(self):
        n = 500
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
        JD = JacobiBandDiagonalization(A, bandwidth = n // 3)
        print "solving large A"
        (B, R) = JD(threshold=1e-16, maxIter=10, printInfo=True, sort=True)
        try:
            self.assert_( N.allclose(B, N.diag(N.arange(n, dtype=N.float)/n), 1e-15, 1e-14))
        except AssertionError:
            D = B-N.diag(N.arange(n, dtype=N.float)/n)
            print D.min()
            print D.max()
            print N.unravel_index(D.argmin(), B.shape)
            print N.unravel_index(D.argmax(), B.shape)
            set_trace()
        self.assert_( N.allclose(N.dot(R.T, R), N.identity(n), 1e-14, 1e-14) )

unittest.main()

