# winak
# This file is Copyright Daniel Strobusch
#

from pysparse import spmatrix 
from winak.curvilinear.numeric import *
LLMatType=type(spmatrix.ll_mat_sym(1,1))
SSSMatType=type(spmatrix.ll_mat_sym(1,1).to_sss())
CSRMatType=type(spmatrix.ll_mat_sym(1,1).to_csr())

class spmatrixIterator:
  def __init__(self, matrix):
    self.matrix = matrix
    if type(self.matrix)==SSSMatType:
      self.diag, self.val, self.col, self.ind = matrix.matrices()
      self.n = len(self.diag)
      self.nod = len(self.val)
      self.nnz = self.n + self.nod
      self.__iter__ = self.iterSSS
    elif type(self.matrix)==LLMatType:
      self.root, self.link, self.col, self.val = matrix.matrices()
      self.n = len(self.root)
      self.nnz = len(self.val)
      self.__iter__ = self.iterLL
    elif type(self.matrix)==CSRMatType:
      self.val, self.col, self.ind = matrix.matrices()
      self.nnz = len(self.val)
      self.__iter__ = self.iterCSR

  def iterLL(self):
    from itertools import izip
    v, i, j = self.matrix.COO()
    return izip(i,j,v)

  def iterCSR(self):
    i = 0
    a = self.ind[0]
    for b in self.ind[1:]:
      for k in range(a, b):
        yield i, self.col[k], self.val[k]
      a = b
      i += 1

  def iterSSS(self):
    i = 0
    for v in self.diag:
      if v!= 0:
        yield i, i, v
      i += 1
    i = 0
    a = self.ind[0]
    for b in self.ind[1:]:
      for k in range(a, b):
        yield i, self.col[k], self.val[k]
      a = b
      i += 1
