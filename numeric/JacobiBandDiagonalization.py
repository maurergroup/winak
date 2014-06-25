from thctk.numeric import *

class JacobiBandDiagonalization:
    """
    This class provides a diagonalization procedure for real symmetric 
    band matrices (in full format only at the moment).
    """

    def __len__(self):
        return self.n

    def __init__(self, A, R = None, bandwidth = 10):
        """
        A is a real symmetric band matrix (as numpy ndarray). The bandwidth (bw)
        is defines the band within the diagonalization should be performed.
        The rotation matrix R will be the eigenvector matrix and will be 
        initialized as identity matrix if R is None. A matrix can be given
        if A is already prediagonalized.
        Because of the algorithm bw can't be larger than n divided by three. 
        """
        self.A0 = A # save the original matrix
        self.A = A.copy() # we will work on a copy
        self.n = n = len(A) 
        bwA = bandwidth # bandwidth of A
        bwR = 0 # bandwidth of R
        bwMax = n // 3 # bwMax is maximal bandwidth
        # do some checks on the bandwidth
        assert bwA >= 1
        if bwA > bwMax:
            bwA = bwMax
            print("Setting bandwidth to maximum of %i." %bwMax)
        self.bwA = bwA
        self.bwMax = bwMax
        if R is None:
            self.R = N.identity(n, dtype = A.dtype)
        else:
            assert A.shape[0] == R.shape[0]
            assert A.shape[1] == R.shape[1]
            self.R = R
            bwR = self._getBandwidth(R)
        if bwR < bwA:
            bwR = bwA
        else:
            bwR = bwR
        self.bwR = bwR
        self.tmp = tuple(N.empty(n) for i in range(3))

    def __call__(self, threshold = 1.0e-8, maxIter = 20, 
                       printInfo = False, sort = False):
        """
        Call, to start the diagonalization procedure.
        WARNING: This Python version is not as numerically stable as the Cython                  Version. Of course, it's also much slower. So don't use it 
                 for serious computations.
        The parameters are:
            'threshold'     defines the maximal magnitude for the off-diagonal 
                            elements inside the defined band. 
            'maxIter'       maximal number of iterations (sweeps).
            'printInfo'     print some infos.
            'sort'          sort the matrices A and R according to the diagonal 
                            of A (eigenvalues)

        To understand the algorithm, take a look at the following picture and 
        note:
        d are the diagonal elements and o the off-diagonal elements. One
        rotation will be applied on rows and columns i an j, were we have to 
        make sure that i < j => li < lj and ri < rj. li and lj denote the left
        boundaries of the band.
        All elements of A outside the band won't be updated durring
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
        self.threshold = threshold # maximal off-diagonal element
        A = self.A # to be diagonalized
        R = self.R # rotation matrix
        n = self.n # dimension of A and R
        (tmp1, tmp2, tmp3) = self.tmp
        bwA = self.bwA # bwA is bandwidth A
        bwR = self.bwR # bwR is bandwidth R
        bwMax = self.bwMax # maximal bandwidth of A and R

        Awidth = 2*bwA + 1
        Rwidth = 2*bwR + 1

        converged = False
        iter = 0 # sweep counter
        for sweep in [self,]*maxIter:
            iter += 1 # count sweeps
            converged = True # assume convergence 
            for (i,j) in sweep:
                # we have to rotate again, a long as range(p) is not empty
                # i.e. no convergence at the moment
                converged = False

                # compute rotation angle theta 

                # (see Numerical Recipes in C p.467f)
                g = N.abs(A[i,j])*100.0
                h = A[j,j] - A[i,i]
                # detect overflow
                if ( float(N.abs(A[i,i]) + g) == float(N.abs(A[i,i])) and
                     float(N.abs(A[j,j]) + g) == float(N.abs(A[j,j])) ):
                    A[i,j] = 0.0
                    continue # skip the rotation
                elif float(N.abs(h) + g) == float(N.abs(h)):
                    t = A[i,j] / h
                else:
                    theta = 0.5 * h / A[i,j]
                    t = 1.0 / (N.abs(theta) + N.sqrt(1.0 + theta*theta))
                    if theta < 0.0:
                        t = -t
                c = 1.0/N.sqrt(1+t*t)
                s = -t*c # WARNING: for python code need "-s" for cython "+s"
                # get boundaries on rows (columns) i and j
                li = i - bwA
                ri = i + bwA + 1
                lj = j - bwA
                rj = j + bwA + 1
                # assert not (li < 0 and ri > n)
                # assert not (lj < 0 and rj > n)
                # assert N.abs((c*c - s*s)*A[i,j] + s*c*(A[i,i]-A[j,j])) < 1.0e-15
                if lj <= 0: # highest corner of matrix (very small i,j)
                    # assert li < 0
                    # li = lj = 0
                    rji = rj - ri

                    a1 = N.multiply(A[:ri,i],  c, tmp1[:ri])
                    a2 = N.multiply(A[:ri,i], -s, tmp2[:ri])
                    a3 = N.multiply(A[:rj,j],  s, tmp3[:rj])
                    N.add(a1, a3[:-rji], A[:ri,i])

                    a3 = N.multiply(A[:rj,j],  c, a3)
                    N.add(a2, a3[:-rji], A[:ri,j])
                    A[ri:rj,j] = a3[-rji:]

                    a1 = N.multiply(A[i,:ri],  c, a1)
                    a2 = N.multiply(A[i,:ri], -s, a2)
                    a3 = N.multiply(A[j,:rj],  s, a3)
                    N.add(a1, a3[:-rji], A[i,:ri])

                    a3 = N.multiply(A[j,:rj],  c, a3)
                    N.add(a2, a3[:-rji], A[j,:ri])
                    A[j,ri:rj] = a3[-rji:]

                elif li < 0: # higher corner of matrix (small i,j)
                    # assert lj >= 0
                    # li = 0
                    rji = rj - ri
                    rlj = rj - lj

                    a1 = N.multiply(A[:ri,i],  c, tmp1[:ri])
                    a2 = N.multiply(A[:ri,i], -s, tmp2[:ri])
                    a3 = N.multiply(A[lj:rj,j],  s, tmp3[:rlj])
                    N.add(a1[lj:], a3[:-rji], A[lj:ri,i])
                    A[:lj,i] = a1[:lj]

                    a3 = N.multiply(A[lj:rj,j],  c, a3)
                    N.add(a2[lj:], a3[:-rji], A[lj:ri,j])
                    A[ri:rj,j] = a3[-rji:]

                    a1 = N.multiply(A[i,:ri],  c, a1)
                    a2 = N.multiply(A[i,:ri], -s, a2)
                    a3 = N.multiply(A[j,lj:rj],  s, a3)
                    N.add(a1[lj:], a3[:-rji], A[i,lj:ri])
                    A[i,:lj] = a1[:lj]

                    a3 = N.multiply(A[j,lj:rj],  c, a3)
                    N.add(a2[lj:], a3[:-rji], A[j,lj:ri])
                    A[j,ri:rj] = a3[-rji:]
                # since i < j the order of the checks matters
                elif ri >= n: # lowest corner of matrix (very large i,j)
                    # assert rj > n # if ri > n also rj > n since ri > rj
                    # ri = rj = n
                    lji = lj - li
                    rli = n - li
                    rlj = n - lj
                    # rli > rlj
                    # assert rli > rlj
                    a1 = N.multiply(A[li:,i],  c, tmp1[:rli])
                    a2 = N.multiply(A[li:,i], -s, tmp2[:rli])
                    a3 = N.multiply(A[lj:,j],  s, tmp3[:rlj])
                    N.add(a1[lji:], a3, A[lj:,i])
                    A[li:lj,i] = a1[:lji]

                    a3 = N.multiply(A[lj:,j],  c, a3)
                    N.add(a2[lji:], a3, A[lj:,j])

                    a1 = N.multiply(A[i,li:],  c, a1)
                    a2 = N.multiply(A[i,li:], -s, a2)
                    a3 = N.multiply(A[j,lj:],  s, a3)
                    N.add(a1[lji:], a3, A[i,lj:])
                    A[i,li:lj] = a1[:lji]

                    a3 = N.multiply(A[j,lj:],  c, a3)
                    N.add(a2[lji:], a3, A[j,lj:])
                elif rj > n: # lower corner of matrix (large i,j)
                    # assert ri <= n # only rj > n
                    # rj = n
                    lji = lj - li
                    rji = n - ri
                    rli = ri - li
                    rlj = n - lj
                    # rli > rlj
                    # assert rli > rlj
                    a1 = N.multiply(A[li:ri,i],  c, tmp1[:rli])
                    a2 = N.multiply(A[li:ri,i], -s, tmp2[:rli])
                    a3 = N.multiply(A[lj:,j],  s, tmp3[:rlj])
                    N.add(a1[lji:], a3[:-rji], A[lj:ri,i])
                    A[li:lj,i] = a1[:lji]

                    a3 = N.multiply(A[lj:,j],  c, a3)
                    N.add(a2[lji:], a3[:-rji], A[lj:ri,j])
                    A[ri:,j] = a3[-rji:]

                    a1 = N.multiply(A[i,li:ri],  c, a1)
                    a2 = N.multiply(A[i,li:ri], -s, a2)
                    a3 = N.multiply(A[j,lj:],  s, a3)
                    N.add(a1[lji:], a3[:-rji], A[i,lj:ri])
                    A[i,li:lj] = a1[:lji]

                    a3 = N.multiply(A[j,lj:],  c, a3)
                    N.add(a2[lji:], a3[:-rji], A[j,lj:ri])
                    A[j,ri:] = a3[-rji:]
                else: # middle part of matrix 
                    # i < j => li < lj
                    lji = lj - li
                    rji = rj - ri
                    a1 = N.multiply(A[li:ri,i],  c, tmp1[:Awidth])
                    a2 = N.multiply(A[li:ri,i], -s, tmp2[:Awidth])
                    a3 = N.multiply(A[lj:rj,j],  s, tmp3[:Awidth])
                    N.add(a1[lji:], a3[:-rji], A[lj:ri,i])
                    A[li:lj,i] = a1[:lji]

                    a3 = N.multiply(A[lj:rj,j],  c, a3)
                    N.add(a2[lji:], a3[:-rji], A[lj:ri,j])
                    A[ri:rj,j] = a3[-rji:]

                    a1 = N.multiply(A[i,li:ri],  c, a1)
                    a2 = N.multiply(A[i,li:ri], -s, a2)
                    a3 = N.multiply(A[j,lj:rj],  s, a3)
                    N.add(a1[lji:], a3[:-rji], A[i,lj:ri])
                    A[i,li:lj] = a1[:lji]

                    a3 = N.multiply(A[j,lj:rj],  c, a3)
                    N.add(a2[lji:], a3[:-rji], A[j,lj:ri])
                    A[j,ri:rj] = a3[-rji:]

                if bwR > bwMax:
                    a1 = N.multiply(R[:,i],  c, tmp1)
                    a2 = N.multiply(R[:,i], -s, tmp2)
                    a3 = N.multiply(R[:,j],  s, tmp3)
                    N.add(a1, a3, R[:,i])
                    a3 = N.multiply(R[:,j],  c, tmp3)
                    N.add(a2, a3, R[:,j])
                else:
                    li = i - bwR
                    ri = i + bwR + 1
                    lj = j - bwR
                    rj = j + bwR + 1
                    if lj <= 0:
                        li = lj = 0
                        rji = rj - ri

                        a1 = N.multiply(R[:ri,i],  c, tmp1[:ri])
                        a2 = N.multiply(R[:ri,i], -s, tmp2[:ri])
                        a3 = N.multiply(R[:rj,j],  s, tmp3[:rj])
                        N.add(a1, a3[:-rji], R[:ri,i])
                        R[ri:rj,i] = a3[-rji:]

                        a3 = N.multiply(R[:rj,j],  c, a3)
                        N.add(a2, a3[:-rji], R[:ri,j])
                        R[ri:rj,j] = a3[-rji:]
                    elif li < 0:
                        li = 0
                        rji = rj - ri
                        rlj = rj - lj

                        a1 = N.multiply(R[:ri,i],  c, tmp1[:ri])
                        a2 = N.multiply(R[:ri,i], -s, tmp2[:ri])
                        a3 = N.multiply(R[lj:rj,j],  s, tmp3[:rlj])
                        N.add(a1[lj:], a3[:-rji], R[lj:ri,i])
                        R[:lj,i] = a1[:lj]
                        R[ri:rj,i] = a3[-rji:]

                        a3 = N.multiply(R[lj:rj,j],  c, a3)
                        N.add(a2[lj:], a3[:-rji], R[lj:ri,j])
                        R[ri:rj,j] = a3[-rji:]
                        R[:lj,j] = a2[:lj]
                    elif ri >= n:
                        ri = rj = n
                        lji = lj - li
                        rli = n - li
                        rlj = n - lj

                        a1 = N.multiply(R[li:,i],  c, tmp1[:rli])
                        a2 = N.multiply(R[li:,i], -s, tmp2[:rli])
                        a3 = N.multiply(R[lj:,j],  s, tmp3[:rlj])
                        N.add(a1[lji:], a3, R[lj:,i])
                        R[li:lj,i] = a1[:lji]

                        a3 = N.multiply(R[lj:,j],  c, a3)
                        N.add(a2[lji:], a3, R[lj:,j])
                        R[li:lj,j] = a2[:lji]
                    elif rj > n: 
                        rj = n
                        lji = lj - li
                        rji = n - ri
                        rli = ri - li
                        rlj = n - lj
                         
                        a1 = N.multiply(R[li:ri,i],  c, tmp1[:rli])
                        a2 = N.multiply(R[li:ri,i], -s, tmp2[:rli])
                        a3 = N.multiply(R[lj:,j],  s, tmp3[:rlj])
                        N.add(a1[lji:], a3[:-rji], R[lj:ri,i])
                        R[li:lj,i] = a1[:lji]
                        R[ri:,i] = a3[-rji:]

                        a3 = N.multiply(R[lj:,j],  c, a3)
                        N.add(a2[lji:], a3[:-rji], R[lj:ri,j])
                        R[ri:,j] = a3[-rji:]
                        R[li:lj,j] = a2[:lji]
                    else:
                        lji = lj - li
                        rji = rj - ri

                        a1 = N.multiply(R[li:ri,i],  c, tmp1[:Rwidth])
                        a2 = N.multiply(R[li:ri,i], -s, tmp2[:Rwidth])
                        a3 = N.multiply(R[lj:rj,j],  s, tmp3[:Rwidth])
                        N.add(a1[lji:], a3[:-rji], R[lj:ri,i])
                        R[li:lj,i] = a1[:lji]
                        R[ri:rj,i] = a3[-rji:]

                        a3 = N.multiply(R[lj:rj,j],  c, a3)
                        N.add(a2[lji:], a3[:-rji], R[lj:ri,j])
                        R[ri:rj,j] = a3[-rji:]
                        R[li:lj,j] = a2[:lji]

                    # no boundary checks needed here
                    tmp = R[ri:rj,i].nonzero()[0]
                    if len(tmp) == 0:
                        k = 0
                    else:
                        k = tmp[-1] - i + 1
                    tmp = R[li:lj,j].nonzero()[0]
                    if len(tmp) == 0:
                        l = 0
                    else:
                        l = i - tmp[0] - 1
                    m = max(k, l)
                    bwR += m
                    Rwidth += 2*m
            if converged: 
                break
        if iter >= maxIter and not converged:
            raise ValueError('No convergence!')
        if printInfo:
            print "Needed %i iterations." %iter
        # update the bandwidth of R (for later inspection by the user)
        if bwR > bwMax:
            self.bwR = self._getBandwidth(R)
        else:
            self.bwR = bwR
        if sort:
            self._sort()
        # assert N.allclose( N.dot(R.T, R), N.identity(n), 1e-15, 1e-14)
        # Now calculate the accurate matrix resulting from the rotations
        A = N.dot(R.T, N.dot(self.A0, R))
        return (A, R) # return eigenvectors as columns

    def __iter__(self):
        n = self.n
        A = self.A
        bwA = self.bwA
        threshold = self.threshold
        # this array will be sorted to get the elements to be rotated 
        # in correct order
        sortMe = N.empty(n, dtype = N.dtype( [('a', N.float), ('i', N.int)], 
                           align = True))
        r = N.arange(n, dtype = N.float)
        for k in xrange(1, bwA + 1):
            Aij = N.abs(N.diag(A, k))
            mask = Aij >= threshold
            part = n-k
            sortMe['a'][:part] = -Aij
            sortMe['i'][:part] = r[:part]
            q = sortMe[mask]
            q[:part].sort(order = ['a',])
            # Aij = N.asarray([Aij, r[:-k]]).T
            # Aij = Aij[mask].tolist()
            # Aij.sort(reverse = True)
            for i in q['i'][:part]:
                yield (i, i + k)

    def _getBandwidth(self, M):
        """
        Calculated bandwidth.
        """
        bw = 0
        n = len(M)
        assert M.shape == M.T.shape
        for (Mi, i) in zip(M.T, range(n)):
            nz = Mi.nonzero()[0]
            if len(nz) > 0:
                bw = max(bw, max(i - nz[0], nz[-1] - i))
        return bw

    def _sort(self):
        n = self.n
        A = self.A
        R = self.R
        l = zip(N.diag(A), range(n))
        l.sort()
        p = zip(*l)[1]
        N.take(A, p, axis = 0, out = A)
        N.take(A, p, axis = 1, out = A)
        N.take(R, p, axis = 1, out = R)

