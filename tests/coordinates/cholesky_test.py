from scikits.sparse.cholmod import cholesky
import numpy as np
from scipy.sparse import csc_matrix
nmatrix = 5
A = np.eye(nmatrix)*10
A[3,1] = 4
A[1,3] = 4
#A[5,4] = 2
#A[4,5] = 2

print A

E, VV = np.linalg.eig(A)
print E
print VV
AA = csc_matrix(A)

Ls = []

factor = cholesky(AA)
L = factor.L()

for i in range(50):
    #print i
    factor = cholesky(AA)
    L = factor.L()
    AA = L.conjugate().transpose().dot(L)
    Ls.append(L.conjugate().transpose())

print np.diag(AA.todense())
LL = np.zeros_like(Ls[0].todense())
LL = Ls[0].todense()
for L in Ls:
    LL = np.dot(L.todense(),LL)

print AA.todense()
V = np.linalg.inv(LL)
#print V
for i in range(nmatrix):
    vec = np.array(V[:,i])
    norm = np.linalg.norm(vec)
    V[:,i] =  vec/norm
print V

print np.dot(np.linalg.inv(V),np.dot(A,V))
