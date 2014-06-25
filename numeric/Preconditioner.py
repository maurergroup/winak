# thctk.numeric.Preconditioner
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2006 Christoph Scheurer
#
#   This file is part of thctk.
#
#   thctk is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   thctk is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#   This file is also:
#   Copyright (C) 2006 Mehdi Bounouar

"""
Preconditioner Module

The goal of preconditioning is to find a matrix K with:
1.  K is a good approximation of the matrix (A - sig * M) in some sense
2.  The system K.x = b can be solved much faster than the original system
3.  The cost for constructing K is smaller than the benefit gained by using
    the preconditioner.

N.B: The use of a preconditioner pays off only if it reduces the number of
    iterations significantely.
"""

from thctk.numeric import *
from thctk.numeric._numeric import inv_L_x_cd, inv_Lt_x_cd, inv_LtL_x_cd, inv_LU_sq_x
from thctk.numeric.SparseMatrix import CSRd, CSR, largeLL
from thctk.numeric.icfs import dicfs
#from thctk.numeric.sparslab import ainvsr2module, rifsrmodule 
from warnings import warn
from pysparse import precon, spmatrix
import itertools
from thctk.numeric.ilut import ilu0,milu0,ilut,ilud,ilutp,iludp,iluk,lutsol,lusol
from thctk.numeric.csrVSmsr import msrcsr, csrmsr
import copy
import os
import time
from thctk.QD import VCI

comb = VCI.comb
LA=importLinearAlgebra()
arrayType = type(N.array(()))
LLMatType = type(spmatrix.ll_mat_sym(1,1))
SSSMatType = type(spmatrix.ll_mat_sym(1,1).to_sss())
CSRMatType = type(spmatrix.ll_mat_sym(1,1).to_csr())

PreconditionerList = ['None', 'Jacobi', 'SSOR', 'Chol', 'ILU0', 'MILU0', 
    'ILUT', 'ILUTP', 'ILUD', 'ILUDP', 'ILUK', 'LSP', 'CHP', 'INV', 'MSPAI']

class Preconditioner:
    def __init__ (self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def precon (self, x,y):
        # apply the precon method to x  and stores y 
        N.multiply((x, self.matrix, y))

class PolynomialPreconditioner:
    def __init__(self, A, polynomialDegree = 4, alpha = 1., beta = 0., norm = True):
      self.LL = largeLL(A)
      self.K = self.LL.to_CSR()
      self.shape = self.K.shape
      self.nnz = self.K.nnz
      n = self.K.n
      self.alpha, self.beta = alpha, beta
      self.polynomialDegree = polynomialDegree +1
      self.s = self.polynomial()
      self.coef = self.s[polynomialDegree]
      if norm: self.normalization()
      self.xt = N.zeros(n, nxFloat)
      self.xk = N.zeros(n, nxFloat)  

    def coefficients(self):
      return self.coef

    def precon(self, x, y):
      xk, xt = self.xk, self.xt
      K = self.K
      coef = self.coef
      y  = N.multiply(x, coef[0], y)

      K.matvec(x, xt) 
      y += N.multiply(xt, coef[1], xk)
      xk, xt = xt, xk
      for i in range(2, len(coef)):
        K.matvec(xk, xt)
        y += N.multiply(xt, coef[i], xk)
        xk, xt = xt, xk
# axpy

    def normalization(self):
      s = self.s
      for n in s:
        Norm = N.sqrt(N.sum(s[n]**2))
        s[n] /= Norm

class combinedPreconditioner:
    def __init__ (self, block1, block2, odBlock, coef, order=1, ld = True):
      self.block1 = block1
      self.block2 = block2
      n = block1.n
      m = block2.n
      self.m = m
      self.n = n
      self.shape = (n+m,n+m)
      self.yt = N.zeros((n+m), nxFloat)
      if ld:
        self.odp=outerDiagonalPrec(block1,block2, ld=odBlock, coef=coef, order=order)
      else:
        self.odp=outerDiagonalPrec(block1,block2, ud=odBlock, coef=coef, order=order)
# self.pblock1 = Preconditioner(block1)

    def precon(self, x, y):
      yt = self.yt
      n  = self.n
      y1 = yt[:n]
      y2 = yt[n:]
      x1 = x[:n]
      x2 = x[n:]
      self.odp.precon(x, y)
      self.pblock1.precon(x1, y1)
      self.pblock2.precon(x2, y2)
      y[:n] += y1
      y[n:] += y2

class outerDiagonalPrec:
    def __init__(self, d1, d2, ld = None, ud = None, coef = None, order = 1):
      self.coef = coef
      self.d1 = d1.to_CSR()
      self.d2 = d2.to_CSR()
      self.n = d1.n
      self.m = d2.n
      n = self.n
      m = self.m
      if ld != None and ud == None:
        self.ld = ld.to_CSR()
        ud = self.ld.transpose() 
        self.ud = ud
      elif ud != None and ld == None:
        self.ud = ud.to_CSR()
        ld = self.ud.transpose() 
        self.ld = ld
      else: raise 'outerdiagonal block is missing!'
      self.yt = N.zeros((n+m), nxFloat)
      self.yk = N.zeros((n+m), nxFloat)
      self.xt = N.zeros((n+m), nxFloat)
      self.xk = N.zeros((n+m), nxFloat)
      self.xi = N.zeros((n+m), nxFloat)
      self.xj = N.zeros((n+m), nxFloat)
      self.xh = N.zeros((n+m), nxFloat)
      self.xl = N.zeros((n+m), nxFloat)
      if   order == 1:
        self.precon = self.precon1st
      elif order == 2:
        self.precon = self.precon2nd
      elif order == 3:
        self.precon = self.precon3rd
      elif order == 4:
        self.precon = self.precon4th

    def firstOrderI(self, x, y):
      d1 = self.d1
      d1.matvec(x,y)
      
    def firstOrderII(self, x, y):
      ud = self.ud
      ud.matvec(x,y)

    def firstOrderIII(self, x, y):
      ld = self.ld
      ld.matvec(x,y)
      
    def firstOrderIV(self, x, y):
      d2 = self.d2
      d2.matvec(x,y)

    def firstOrder(self, x, y):
      ud, ld = self.ud, self.ld
      n, yt = self.n, self.yt
      x1  = x[:n]
      x2  = x[n:]
      y1  = yt[:n]
      y2  = yt[n:]
      ud.matvec(x2, y1)
      ld.matvec(x1, y2)
      y[:n] = y1
      y[n:] = y2
  
    def secondOrderI(self, x, y):
      d1, ud, ld = self.d1, self.ud, self.ld
      n, xk, xt = self.n, self.xk, self.xt
      xk1 = xk[:n]
      xt1 = xt[:n]
      xt2 = xt[n:]
      ld.matvec(x, xt2)
      ud.matvec(xt2, xk1)
      y[:] = xk1
      d1.matvec(x, xt1)
      d1.matvec(xt1, xk1)
      y += xk1

    def secondOrderII(self, x, y):
      d1, d2, ud = self.d1, self.d2, self.ud
      n, xk, xt = self.n, self.xk, self.xt
      xk1 = xk[:n]
      xt1 = xt[:n]
      xt2 = xt[n:]
      d2.matvec(x, xt2) 
      ud.matvec(xt2, xk1)
      y[:] = xk1
      ud.matvec(x, xt1)
      d1.matvec(xt1, xk1) 
      y += xk1

    def secondOrderIII(self, x, y):
      d1, d2, ld = self.d1, self.d2, self.ld
      n, xk, xt = self.n, self.xk, self.xt
      xk2 = xk[n:]
      xt1 = xt[:n]
      xt2 = xt[n:]
      ld.matvec(x, xt2)
      d2.matvec(xt2, xk2) 
      y[:] = xk2
      d1.matvec(x, xt1) 
      ld.matvec(xt1, xk2)
      y += xk2

    def secondOrderIV(self, x, y):
      d2, ud, ld = self.d2, self.ud, self.ld
      n, xk, xt = self.n, self.xk, self.xt
      xk1 = xk[:n]
      xk2 = xk[n:]
      xt1 = xt[:n]
      xt2 = xt[n:]
      d2.matvec(x, xt2)
      d2.matvec(xt2, xk2)
      y[:] = xk2
      ud.matvec(x, xt1)
      ld.matvec(xt1, xk2)
      y += xk2

    def thirdOrderI(self, x, y):
      d1, ld = self.d1, self.ld
      n, xi, xj = self.n, self.xi, self.xj
      xi1 = xi[:n]
      xi2 = xi[n:]
      xj1 = xj[:n]
      ld.matvec(x, xi2)
      self.secondOrderII(xi2, xj1)
      y[:] = xj1 
      d1.matvec(x, xi1)
      self.secondOrderI(xi1, xj1)
      y += xj1 

    def thirdOrderII(self, x, y):
      d2, ud = self.d2, self.ud
      n, xi, xj = self.n, self.xi, self.xj
      xi1 = xi[:n]
      xi2 = xi[n:]
      xj1 = xj[:n]
      d2.matvec(x, xi2)
      self.secondOrderII(xi2, xj1)
      y[:] = xj1
      ud.matvec(x, xi1)
      self.secondOrderI(xi1, xj1)
      y += xj1 

    def thirdOrderIII(self, x, y):
      d1, ld = self.d1, self.ld
      n, xi, xj = self.n, self.xi, self.xj
      xi1 = xi[:n]
      xi2 = xi[n:]
      xj2 = xj[n:]
      ld.matvec(x, xi2)
      self.secondOrderIV(xi2, xj2)
      y[:] = xj2 
      d1.matvec(x, xi1)
      self.secondOrderIII(xi1, xj2)
      y += xj2 

    def thirdOrderIV(self, x, y):
      d2, ud = self.d2, self.ud
      n, xi, xj = self.n, self.xi, self.xj
      xi1 = xi[:n]
      xi2 = xi[n:]
      xj2 = xj[n:]
      d2.matvec(x, xi2)
      self.secondOrderIV(xi2, xj2)
      y[:] = xj2 
      ud.matvec(x, xi1)
      self.secondOrderIII(xi1, xj2)
      y += xj2 

    def fourthOrderI(self, x, y):
      d1, ld = self.d1, self.ld
      n, xh, xl = self.n, self.xh, self.xl
      xh1 = xh[:n]
      xl1 = xl[:n]
      xl2 = xl[n:]
      ld.matvec(x, xl2)
      self.thirdOrderII(xl2, xh1)
      y[:] = xh1 
      d1.matvec(x, xl1)
      self.thirdOrderI(xl1, xh1)
      y += xh1 

    def fourthOrderII(self, x, y):
      d2, ud = self.d2, self.ud
      n, xh, xl = self.n, self.xh, self.xl
      xh1 = xh[:n]
      xl1 = xl[:n]
      xl2 = xl[n:]
      d2.matvec(x, xl2)
      self.thirdOrderII(xl2, xh1)
      y[:] = xh1 
      ud.matvec(x, xl1)
      self.thirdOrderI(xl1, xh1)
      y += xh1 

    def fourthOrderIII(self, x, y):
      d1, ld = self.d1, self.ld
      n, xh, xl = self.n, self.xh, self.xl
      xh2 = xh[n:]
      xl1 = xl[:n]
      xl2 = xl[n:]
      ld.matvec(x, xl2)
      self.thirdOrderIV(xl2, xh2)
      y[:] = xh2 
      d1.matvec(x, xl1)
      self.thirdOrderIII(xl1, xh2)
      y += xh2 

    def fourthOrderIV(self, x, y):
      d2, ud = self.d2, self.ud
      n, xh, xl = self.n, self.xh, self.xl
      xh2 = xh[n:]
      xl1 = xl[:n]
      xl2 = xl[n:]
      d2.matvec(x, xl2)
      self.thirdOrderIV(xl2, xh2)
      y[:] = xh2 
      ud.matvec(x, xl1)
      self.thirdOrderIII(xl1, xh2)
      y += xh2 

    def precon1st(self, x, y):
      yk, yt = self.yk, self.yt
      coef = self.coef
      y  = N.multiply(x, coef[0], y)
      self.firstOrder(x, yt) 
      y += N.multiply(yt, coef[1], yk)

    def precon2nd(self, x, y):
      n, yk, yt = self.n, self.yk, self.yt
      coef = self.coef
      y  = N.multiply(x, coef[0], y)
      self.firstOrder(x, yt) 
      y += N.multiply(yt, coef[1], yk)
      self.secondOrderII(x[n:], yt[:n]) 
      self.secondOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[2], yk)

    def precon3rd(self, x, y):
      n, yk, yt = self.n, self.xi, self.yt
      coef = self.coef
      y  = N.multiply(x, coef[0], y)
      self.firstOrder(x, yt) 
      y += N.multiply(yt, coef[1], yk)
      self.secondOrderII(x[n:], yt[:n]) 
      self.secondOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[2], yk)
      self.thirdOrderII(x[n:], yt[:n]) 
      self.thirdOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[3], yk)

    def precon4th(self, x, y):
      n, yk, yt = self.n, self.xi, self.yt
      coef = self.coef
      y  = N.multiply(x, coef[0], y)
      self.firstOrder(x, yt) 
      y += N.multiply(yt, coef[1], yk)
      self.secondOrderII(x[n:], yt[:n]) 
      self.secondOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[2], yk)
      self.thirdOrderII(x[n:], yt[:n]) 
      self.thirdOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[3], yk)
      self.fourthOrderII(x[n:], yt[:n]) 
      self.fourthOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[4], yk)

    def justPrecon(self, x, y):
      n, yk, yt = self.n, self.xi, self.yt
      coef = self.coef
      y  = N.multiply(x, coef[0], y)
    
      self.firstOrderI(x[:n], yt[:n]) 
      self.firstOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[1], yk)
      self.firstOrderII(x[n:], yt[:n]) 
      self.firstOrderIV(x[n:], yt[n:]) 
      y += N.multiply(yt, coef[1], yk)

      self.secondOrderI(x[:n], yt[:n]) 
      self.secondOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[2], yk)
      self.secondOrderII(x[n:], yt[:n]) 
      self.secondOrderIV(x[n:], yt[n:]) 
      y += N.multiply(yt, coef[2], yk)

      self.thirdOrderI(x[:n], yt[:n]) 
      self.thirdOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[3], yk)
      self.thirdOrderII(x[n:], yt[:n]) 
      self.thirdOrderIV(x[n:], yt[n:]) 
      y += N.multiply(yt, coef[3], yk)

      self.fourthOrderI(x[:n], yt[:n]) 
      self.fourthOrderIII(x[:n], yt[n:]) 
      y += N.multiply(yt, coef[4], yk)
      self.fourthOrderII(x[n:], yt[:n]) 
      self.fourthOrderIV(x[n:], yt[n:]) 
      y += N.multiply(yt, coef[4], yk)
 
class leastSquaresPolynomial(PolynomialPreconditioner):
    def weightFunction(self):
      alpha, beta = self.alpha, self.beta
      pd = self.polynomialDegree
      wF = {}
      kalpha = 1 + alpha
      for k in range(1,pd+1):
        R = N.ones((k+1), nxFloat)
        kbeta = k + beta
        div0 = kbeta / kalpha
        for j in range(1,k+1):
          R[j] = comb(k, j) * div0 * (-1)**j
          for i in range(1,j):
            R[j] *= (kbeta - i) / (kalpha + i)
        wF[k] = R
      self.wF = wF
 
    def polynomial(self):
      self.weightFunction()    
      s = {}
      R = self.wF
      for l in R:
        n = len(R[l])
        s[l-1] = N.zeros((n-1), nxFloat)
        for k in range(1,n):
          s[l-1][k-1] += comb((n-1),k)*(-1)**(k-1) 
        for j in range(1,n):
          for k in range(n-j):
            s[l-1][k+j-1] -= R[l][j]*comb((n-j-1),k)*(-1)**(k) 
      return s

class chebyshevPolynomial(PolynomialPreconditioner):
    def cheb1stKind(self):
      pd = self.polynomialDegree
      T = {}
      T[0] = N.array((1,), nxFloat)
      T[1] = N.array((0,1), nxFloat)
      for i in range(2, pd+1):
        T[i] = N.zeros((i+1), nxFloat)
        for j in range(len(T[i-1])):
          T[i][j+1] = 2*T[i-1][j]
        for j in range(len(T[i-2])):
          T[i][j] -= T[i-2][j]
      return T
   
    def weightFunction(self):
      T = self.cheb1stKind()
      alpha, beta = self.alpha, self.beta
      pd = self.polynomialDegree
      wF = {}
      nd = alpha - beta
      dif = (alpha + beta)/nd
      lam = 2/nd
      for k in range(1,pd+1):
        ln = len(T[k])
        R = N.zeros(ln, nxFloat)
        Tka = 0.
        for i in range(ln):
          Tka += T[k][i]*(dif**i)
          for j in range(i+1):
            R[j] += T[k][i]*comb(i,j)*dif**(i-j)*(-2)**j
        R /= Tka
        wF[k] = R
      self.wF = wF

    def polynomial(self):
      self.weightFunction()    
      R = self.wF
      s = {}
      for i in R:
        s[i-1] = -R[i][1:]
      return s

class Inverse(Preconditioner):
    def __init__(self, matrix, issym = None):
      self.LL = largeLL(matrix, issym)
      array = self.LL.to_array2D()
      inv = LA.inverse(array)
      A = largeLL(inv)
      self.matrix = A.to_CSR()    
      self.shape = self.matrix.shape

    def precon(self, x, y):
      K = self.matrix
      K.matvec(x, y)

class Jacobi (Preconditioner):
    """ Jacobi preconditioner from PySparse package
        With the defaults parameters is equivalent to a diagonal preconditioner
    """
    def __init__ (self, matrix, omega=1.0, steps=1):
        self.shape = matrix.shape
        self.omega = omega
        self.steps = steps
        
        if type(matrix) == arrayType:
            matrix = array2_ll(matrix)
            self.matrix = matrix.to_sss(); del matrix
        elif type(matrix) == spmatrix.LLMatType :
            self.matrix = matrix.to_sss(); del matrix
        elif type(matrix) == spmatrix.SSSMatType :
            self.matrix = matrix
        else : raise "Type not recognized"
        self.matrix =  precon.jacobi(self.matrix, self.omega, self.steps)
    
    def precon (self, x,y):
        self.matrix.precon(x,y)
  
class SSOR (Preconditioner):
    """ Symmetric succesive over relaxation (SSOR)
        from PySparse package
    """
    def __init__ (self, matrix, omega=1.0, steps=1):
        self.shape = matrix.shape
        self.omega = omega
        self.steps = steps
        
        if type(matrix) == arrayType:
            matrix = array2_ll(matrix)
            self.matrix = matrix.to_sss(); del matrix
        elif type(matrix) == spmatrix.LLMatType :
            self.matrix = matrix.to_sss(); del matrix
        elif type(matrix) == spmatrix.SSSMatType :
            self.matrix = matrix
        else : raise "Type not recognized"
        self.matrix =  precon.ssor(self.matrix, self.omega, self.steps)

    def precon (self, x, y):
        self.matrix.precon(x,y)

class Cholesky (Preconditioner):
    """ Cholesky preconditioner
    """
    def __init__ (self, matrix, p = 10, eps = 1.0e-6, fout = 1, lambd = 0, offset = 0):
        """
        p: fill in
        lambd : shift
        """
        self.shape = matrix.shape
        n = self.shape[0]
        self.offset = offset
        MT = type(matrix)
        if hasattr(matrix, 'toCSR'):
            self.A = matrix
            self.Annz = len(self.A.x) # number of nonzero elements
        elif MT == LLMatType or MT == SSSMatType or MT == CSRMatType:
          mat = largeLL(matrix)
          self.A = mat.to_CSRd()
          self.Annz = self.A.nnz
        else:
            md, mx, mj, mi = matrix.matrices()
            self.Annz = len(mx)
            self.A = CSRd(len(md), self.Annz, mi, mj, mx, md, type = nxTypecode(mx), offset = self.offset)
        
        info = self.cholesky(eps=eps, p=p, lambd=lambd, fout = 1)
        if info < 0:
            raise ArithmeticError('Incomplete Cholesky decomposition failed: '+ `info`)
    
    def cholesky(self, p = 10, eps = 1.0e-6, fout = 1, lambd = 0):
        A = self.A
        n = len(A.d)
        if lambd > 0:
            d = A.d + lambd
        else:
            d = A.d
        nnz = self.Annz + p*n
        if not hasattr(self, 'L') or len(self.L.x) < nnz:
            self.L = CSRd(n, nnz, type = nxTypecode(A.x))
        L = self.L
        one = N.asarray(1, nxTypecode(L.i))
        if not hasattr(self, 'dbltmp'):
            self.dbltmp = N.zeros(2*n, nxFloat)
            self.inttmp = N.zeros(3*n, nxInt32)
        alpha = eps
        L.x, L.d, L.i, L.j, alpha, info = \
            dicfs(A.x, d, A.i, A.j, L.x, L.d, L.i, L.j,
                  self.inttmp, self.dbltmp[:n], self.dbltmp[n:],
                  p = p, alpha = alpha, offset = 1-A.offset)
        self.nnz = nnz
        self.dicfs_alpha = alpha
        if alpha > eps:
            warn("dicfs needs larger than expected scaling alpha = %f" % alpha)
        if fout:    # 1-based output
            L.offset = 1
        else:
            L.i -= one
            L.j -= one
            L.offset = 0
        return info

    def inverseL(self, r):
        L = self.L
        return inv_L_x_cd(L.x, L.d, L.i, L.j, r, L.offset)

    def inverseLt(self, r):
        L = self.L
        return inv_Lt_x_cd(L.x, L.d, L.i, L.j, r, L.offset)

    def inverseA(self, r):
        L = self.L
        return inv_LtL_x_cd(L.x, L.d, L.i, L.j, r, L.offset)

    def precon (self, x, y = None):
        if y is None:
            self.inverseA(x)
        else:
            y[:] = x
            self.inverseA(y)
    
class LU(Preconditioner):
  """LU preconditioner
  """
  def __init__(self, matrix = None, fill = 20, w=1.0e-6, permtol=0.05,
      alpha=0, perc = 0.2, preconditioner = 'ILUT', DebInfo=False, offset=1):
    self.preconditioner = preconditioner
    self.DebInfo = DebInfo
    self.perc = perc
    self.shape = matrix.shape
    self.n = self.shape[0] 
    self.offset = offset
    self.permtol = permtol
    self.w = w
    self.alpha = alpha
    self.fill = fill
    try:
      self.A = matrix.to_csr()
    except:
      self.A = matrix
    a, ja, ia = self.A.matrices()
#   self.a = a
#   self.ja = ja
#   self.ia = ia
    self.a = a.copy()
    self.ja = ja.copy()
    self.ia = ia.copy()
    if offset:
      self.ja += 1
      self.ia += 1

    if preconditioner   == 'ILUT':
      self.ILUT()
      self.cut()
    elif preconditioner == 'ILUTP':
      self.ILUTP()
      self.cut()
    elif preconditioner == 'ILUD':
      self.ILUD()
      self.cut()
    elif preconditioner == 'ILUK':
      self.ILUK()
      self.cut()
    elif preconditioner == 'ILUDP':
      self.ILUDP()
      self.cut()
    elif preconditioner == 'ILU0':
      self.ILU0()
    elif preconditioner == 'MILU0':
      self.MILU0()
    else: 
      error = "\n"+'right now are following options available:\n'+\
"'ILUT'    : Incomplete LU factorization with dual truncation strategy\n"+\
"'ILUTP'   : ILUT with column  pivoting                               \n"+\
"'ILUD'    : ILU with single dropping + diagonal compensation (~MILUT)\n"+\
"'ILUDP'   : ILUD with column pivoting                                \n"+\
"'ILUK'    : level-k ILU                                              \n"+\
"'ILU0'    : simple ILU(0) preconditioning                            \n"+\
"'MILU0'   : MILU(0) preconditioning "
      raise error
    self.nnz = len(self.alu) - 1 

  def cut(self):
    length = self.ju[-1] - 1
    if self.DebInfo: 
      print "---\n"+self.preconditioner+\
            '\nPercent of the input matrix (set):\t', self.perc, '\n'+\
            'Number of Elements of Precond.:\t\t', len(self.alu), "\n"+\
            'Number of NZ Elements of Precond.:\t',length,'\n'+\
            'Last remaining element:\t\t', self.alu[length-1],'\n'+\
            'First cuted element:\t\t\t', self.alu[length], '\n---'
    self.alu = self.alu[:length]  
    self.jlu = self.jlu[:length] 
 
  def ILUT(self):
    a, ja, ia, w, n = self.a, self.ja, self.ia, self.w, self.n
    fill = self.fill
    perc = self.perc
    iwk = n**2 * perc 
    self.alu,self.jlu,self.ju,self.ierr=ilut(a,ja,ia,lfil=fill,droptol=w,iwk=iwk)

  def ILUD(self):
    a, ja, ia, w, n, alpha = self.a, self.ja, self.ia, self.w, self.n, self.alpha
    fill = self.fill
    perc = self.perc
    iwk = n**2 * perc 
    self.alu, self.jlu, self.ju, self.ierr=ilud(a,ja,ia,alpha,tol=w,iwk=iwk)

  def ILUTP(self):
    a,ja,ia,w,n,permtol = self.a, self.ja, self.ia, self.w, self.n, self.permtol
    fill = self.fill
    mbloc=n
    perc = self.perc
    iwk = n**2 * perc 
    self.alu, self.jlu, self.ju, self.perm, self.ierr = ilutp(a, ja, ia, lfil=fill, droptol=w, permtol=permtol, mbloc=mbloc, iwk=iwk)

  def ILUDP(self):
    a,ja,ia,w,n,permtol,alpha = self.a, self.ja, self.ia, self.w, self.n, self.permtol, self.alpha 
    fill = self.fill
    perc = self.perc
    iwk = n**2 * perc 
    self.alu, self.jlu, self.ju, self.perm, self.ierr = iludp(a, ja, ia, droptol=w, permtol=permtol, iwk=iwk)

  def ILUK(self):
    a,ja,ia,n = self.a, self.ja, self.ia, self.n
    fill = self.fill
    perc = self.perc
    iwk = n**2 * perc 
    self.alu, self.jlu, self.ju, self.ierr=iluk(a, ja, ia, lfil=fill, iwk=iwk)

  def ILU0(self):
    a,ja,ia,n = self.a, self.ja, self.ia, self.n
    self.alu, self.jlu, self.ju, self.ierr = ilu0(a, ja, ia)

  def MILU0(self):
    a,ja,ia,n = self.a, self.ja, self.ia, self.n
    self.alu, self.jlu, self.ju, self.ierr = milu0(a, ja, ia)

  def LUTSOL(self, y, x, alu, jlu, ju):
    return lutsol(y, x, alu, jlu, ju)
    
  def LUSOL(self, y, x, alu, jlu, ju):
    return lusol(y, x, alu, jlu, ju)

  def precon(self, x, y=None):
    alu, jlu, ju = self.alu, self.jlu, self.ju
    if y is None:
      return self.LUSOL(x, x, alu, jlu, ju)
    else:
      y[:] = x
      return self.LUSOL(x, y, alu, jlu, ju)

  def MSR2CSR(self):
    n = self.n
    fill = self.fill
    offset = self.offset
    alu = self.alu
    jlu = self.jlu
    ju = self.ju
    cnt = 0
    while alu[-1] == 0:
      cnt +=1
      alu = alu[:-1]
      jlu = jlu[:-1]
    if self.DebInfo: print "number of not used elements:\t", cnt
    self.cnt = cnt 
    La, Lj, Li = msrcsr(n, alu, jlu)
    while Lj[-1] == 0:
      La = La[:-1]
      Lj = Lj[:-1]
    nnz = len(La)
    self.nnz = nnz 
    if offset:
      Lj -= 1
      Li -= 1
    self.offset = 0
    self.matrix = CSR(n, nnz, Li, Lj, La, offset = 0)
#   del self.alu, self.jlu, self.ju

  def CSR2MSR(self):
    n = self.n
    offset = self.offset
    Lx, Lj, Li = self.matrix.x, self.matrix.j, self.matrix.i
    if not offset:
      Li += 1  
      Lj += 1 
    self.offset = 1 
    alu, jlu = csrmsr(Lx, Lj, Li)
    self.alu, self.jlu = alu, jlu
    self.cut()
    if self.offset:
      Li -= 1
      Lj -= 1

  def LU2L_U(self):
    n = self.n
    hint = int(self.perc * (n**2-n)/2) + n
    L = spmatrix.ll_mat(n, hint)
    U = spmatrix.ll_mat(n, hint)
    Lx, Lj, Li = self.matrix.x, self.matrix.j, self.matrix.i
    for k in range(len(Li)-1): 
      for m in range(Li[k], Li[k+1]): 
        l = Lj[m]
        if k < l: L[l,k] = Lx[m]
        elif k == l:
          U[k,l] = 1.
          L[k,l] = 1/Lx[m]
        elif k > l: U[l,k] = Lx[m]
    self.L = L.to_csr()
    self.U = U.to_csr()
#   del self.matrix 
     
  def LxU(self):
    """
    calculate L*U matrix product, which must be equal to A in
    the case of complete LU decomposition.
    """
    n = self.n
    hint = self.fill * (n+1)
    P = spmatrix.ll_mat(n, hint)
    L, U = self.L, self.U
    for i in range(n):
      for j in range(n):
        for k in range(n):
          try:
            P[i,j] += L[i,k]*U[k,j]
          except:
            P[i,j] = L[i,k]*U[k,j]
    self.LU=P            # L*U ll_mat format
    self.LUs=P.to_csr()  # L*U in csr format
     
  def inverse(self, r):
    L = self.L
    U = self.U
    Lx, Lj, Li = L.matrices()
    Ux, Uj, Ui = U.matrices()
    offset = self.offset
    return inv_LU_sq_x(Lx, Lj, Li, Ux, Uj, Ui, r, offset)

  def preconC(self, x, y = None):
      if y is None:
        return self.inverse(x)
      else:
        y[:] = x
        return self.inverse(y)

class LU_python(Preconditioner):
  def __init__(self, matrix = None, p = 20, w = 1.0e-6, offset=0, 
      preconditioner = 'ILUT'):
    self.shape = matrix.shape
    self.n = self.shape[0] 
    self.offset = offset 
    self.A = matrix
    
    if preconditioner == 'ILUT':
      self.ilut(p,w)
    elif preconditioner == 'ILU0':
      self.ilu0()
    elif preconditioner == 'LU':
      self.lu()
    else: 
      error = "\n"+'right now are following options available:\n'+\
              "'LU'   for complete decomposition\n"+\
              "'ILU0' for ILU0 ;)\n"+\
              "'ILUT' (default) needs p and w as parameter\n"
      raise error

  def lu(self):
    """
    complete LU decomposition 
    """
    A = self.A 
    n = self.n
    hint = n 
    L = spmatrix.ll_mat(n, hint)
    U = spmatrix.ll_mat(n, hint)
    for i in range(n):
      for j in range(n):
        if j < i:
          L[i,j] = A[i,j]
          for k in range(j):
            L[i,j] -= L[i,k]*U[k,j]
        elif j == i:
          U[i,j] = 1.
          L[i,j] = A[i,j]
          for k in range(j):
            L[i,j] -= L[i,k]*U[k,j]
        elif j > i:
          U[i,j] = A[i,j] 
          for k in range(i):
            U[i,j] -= L[i,k]*U[k,j]
          U[i,j] /=  L[i,i]
    self.L = L
    self.U = U

  def ilu0(self):
    """
    incomplete LU0 decomposition
    """
    A = self.A 
    n = self.n
    hint = A.nnz
    L = spmatrix.ll_mat(n, hint)
    U = spmatrix.ll_mat(n, hint)
    for i in range(n):
      for j in range(i):
        if A[i,j] != 0:
          L[i,j] = A[i,j]
          for k in range(j):
            L[i,j] -= L[i,k]*U[k,j]
      U[i,i] = 1.
      L[i,i] = A[i,i]
      for k in range(i):
         L[i,i] -= L[i,k]*U[k,i]
      for j in range(i+1,n):
        if A[i,j] != 0:
          U[i,j] = A[i,j] 
          for k in range(i):
            U[i,j] -= L[i,k]*U[k,j]
          U[i,j] /=  L[i,i]
    self.L = L
    self.U = U

  def ilut(self, p, w):
    """
    incomplete LUT decomposition
    p - maximal fill in of a row (without diagonal elements) in U and L
    w - threshold
    """
    A = self.A 
    n = self.n
    hint = p*(n+1)
    v = N.zeros(n, nxFloat)
    L = spmatrix.ll_mat(n, hint)
    U = spmatrix.ll_mat(n, hint)
    for i in range(n):
      Aii = A[i,i]
      v[:] = 0
      temp = []
      for j in range(n):
        if A[i,j]/A[i,i] > w: 
          v[j] = A[i,j]
      for j in range(i):
          m = 0
          for k in range(j):
            v[j] -= v[k]*U[k,j]
          temp += [abs(v[j]), (v[j], m, j)],

      U[i,i] = 1.
      for k in range(i):
          v[i] -= v[k]*U[k,i]
      L[i,i] = v[i]

      for j in range(i+1,n):
          m = 1
          for k in range(i):
            v[j] -= v[k]*U[k,j]
          v[j] /= v[i]
          temp += [abs(v[j]), (v[j], m, j)],

      if len(temp) > p:
        temp.sort()
        temp[:] = temp[-p:]
      for av,(va,m,k) in temp:
        if m:
          U[i,k] = va 
        else:
          L[i,k] = va
    self.L = L
    self.U = U

  def LxU(self):
    """
    calculate L*U matrix product, which must be equal to A in
    the case of complete LU decomposition.
    """
    n = self.n
    hint = self.A.nnz
    P = spmatrix.ll_mat(n, hint)
    L, U = self.L, self.U
    for i in range(n):
      for j in range(n):
        for k in range(n):
          try:
            P[i,j] += L[i,k]*U[k,j]
          except:
            P[i,j] = L[i,k]*U[k,j]
    self.LU=P            # L*U ll_mat format
    self.LUs=P.to_csr()  # L*U in csr format
  
  def inverse(self, r):
    L = self.L.to_csr()
    U = self.U.to_csr()
    Lx, Lj, Li = L.matrices()
    Ux, Uj, Ui = U.matrices()
    offset = self.offset
    return inv_LU_sq_x(Lx, Lj, Li, Ux, Uj, Ui, r, offset)

  def precon (self, x, y = None):
      if y is None:
        return self.inverse(x)
      else:
        y[:] = x
        return self.inverse(y)

class MSPAI(Preconditioner):
  def __init__(self, matrix = None, name='default.mtx', mname='precond.mtx',
   PRECON="/data/samson/Diplomarbeit/./mspai ", op1=1, op2=1, ep=0., mn=6, 
   ns=3, up=2, cs=0, qr=4, fg=0.3, um=1, schur=0, rho=1.0, ch=0):
    par=' -'+`op1`+' -'+`op2`
    par+=' -ep '+`ep`+' -mn '+`mn`+' -ns '+`ns`+' -cs '+`cs`
    par+=' -qr '+`qr`+' -fg '+`fg`+' -um '+`um`+' -schur '+`schur`
    par+=' -rho '+`rho`+' -ch '+`ch`+' -wp 1'
    self.LL = largeLL(matrix)
    self.LL.to_mtx(name)
    os.system(PRECON+name+par)
    print PRECON+name+par
    k = largeLL()
    k.LL_from_mtx(mname)
    K = k.to_CSR()
    self.shape = K.shape
    self.K = K
    self.nnz = K.nnz
#   os.system('rm '+name+' '+mname)
    self.precon = K.matvec


class SAINV(Preconditioner):
  def __init__(self, matrix = None, p = 100, w = 1.0e-6, offset=1):
    self.shape = matrix.shape
    self.n = self.shape[0]
    self.nnz = matrix.nnz
    self.A = matrix.to_csr()
    a, aj, ai = self.A.matrices()
    self.a = a.copy()
    self.aj = aj.copy()
    self.ai = ai.copy()
    if offset:
      self.aj += 1
      self.ai += 1
    self.sainv()

  def sainv(self, w = 1.0e-6, fillmax = 65):
#   A = self.A #must be in csr format
    n = self.n
    a, aj, ai = self.a, self.aj, self.ai
#   print a, '\n', aj, '\n', ai
    self.p=rifsrmodule
    self.p.rifsr(ia=ai, ja=aj, a=a, msglvl=2)
 
def array2_ll (mat):
    "array2_ll(x), given an array returns an array in ll_format"
    mat = N.array(mat, 'd')
    if N.rank(mat) == 1:
      r = 1
      m = len(mat)
      L = spmatrix.ll_mat(r, m)
      for i in range(r):
         for j in range(m):
            L[i,j] = mat[j]
    else:
      r,m = mat.shape
      L = spmatrix.ll_mat(r, m)
      for i in range(r):
         for j in range(m):
            L[i,j] = mat[i,j]
    return L

def ma2_ll_sym (mat):
    "Numeric array symetric matrix to ll_sym format"
    mat = N.array(mat)
    m,m = mat.shape
    L = spmatrix.ll_mat_sym(m, m)
    for i in range(m):
       for j in range(i+1):
          L[i,j] = mat[i,j]
    return L

def a2_ll_sym (vec):
    """Given a lower triangle part of a sym. matrix as an array
    builds the corresponding Matrix in LL-format
    """
    vec = N.array(vec)
    tot = len(vec)
    m = int((-1 + N.sqrt(1 + 8 * tot)) / 2)
    L = spmatrix.ll_mat_sym(m,m)
    k = 0
    for i in range(m):
       for j in range(i+1):
          L[i,j] = vec[k]
	  k += 1
    return L

def a2ma (vec):
    """
    Given the lower triangle part of a Square Matrix as an array, i.e.
    [a11, a12, a22, a13,..., ann]
    builds the corresponding Matrix
    """
    tot = len(vec)
    m = int((-1 + N.sqrt(1 + 8 * tot)) / 2) # solution of a snd order polynom
    L = N.zeros((m,m), nxFloat)
    i,j = 0, 1
    for k in range(m):
	    L[k,:k+1] = vec[i:j]
            i = j
            j += (k+2)
    return L

def ma2a (mat):
    "Given a Sym. Matrix put L triangle in array"
    mat = N.array(mat)
    m,m = mat.shape
    n = (m*m + m)/2
    L = N.zeros(n, nxFloat)
    k = 0
    for i in range(m):
       for j in range(i+1):
          L[k] = mat[i,j]
	  k +=1
    return L

def One_ll (m):
    "return a square unitary matrix in ll_format"
    One = spmatrix.ll_mat(m,m)
    uno = N.array([1],'d')
    for i in range(m):
       One[i,i] = uno[0]
    return One

###### Test Scripts  ######

if __name__ == "__main__":
    A = N.arange(9) +1
    try:
        A = N.outer(A,A)
    except: # Old Numeric
        A = N.outerproduct(A,A)
    B = ma2a(A)
   
    print array2_ll(A)
    print ma2_ll_sym(A)
    print a2_ll_sym(ma2a(A))
    print a2_ll_sym(B)
    print a2ma(B)
    print One_ll(4)

class LL_Preconditioner(largeLL):
  def initPrec(self, fillin = 20, w = 1.0e-5, 
          permtol = 0., perc = 0.2, alpha = 0.5, beta=-0.5, lambd = 0, alLU = 0.5, 
          prec = 'None', polynomialDegree = 4, DebInfo = False, 
          tmng = False):
    """
    fillin  -
    w       -
    permtol -
    perc    -
    prec    -
    DedInfo -
    """
    if tmng: 
      t0 = time.time()
    self.K = self.choosePrecond( A = self.matrix, w = w, fi = fillin, pt = permtol, 
          pe = perc, all = alLU, la = lambd, al = alpha, be =beta, prec = prec, 
          pDeg = polynomialDegree, DI = DebInfo)
    if tmng: 
      self.pcTime = time.time() - t0
    if self.K != None:
      self.precon = self.K.precon

  def choosePrecond(self, A, w, fi, pt, pe, all, la, al, be, prec, pDeg, DI):
      if   prec == 'None':     return None
      elif prec == 'Jacobi':   return Jacobi(A)
      elif prec == 'SSOR':     return SSOR(A)
      elif prec == 'Chol':     return Cholesky(A, p = fi, eps = w, lambd = la)
  #   elif prec == 'SAINV':    return SAINV(A)
      elif prec == 'ILU0':     return LU(A, preconditioner='ILU0', DebInfo=DI)
      elif prec == 'MILU0':    return LU(A, preconditioner='MILU0', DebInfo=DI)
      elif prec == 'ILUT':     return LU(A, fi, w, pt, all, pe, 'ILUT', DI)
      elif prec == 'ILUTP':    return LU(A, fi, w, pt, all, pe, 'ILUTP', DI)
      elif prec == 'ILUD':     return LU(A, fi, w, pt, all, pe, 'ILUD', DI)
      elif prec == 'ILUDP':    return LU(A, fi, w, pt, all, pe, 'ILUDP', DI)
      elif prec == 'ILUK':     return LU(A, fi, w, pt, all, pe, 'ILUK', DI)
      elif prec == 'LSP':      return leastSquaresPolynomial(A, pDeg, al, be)
      elif prec == 'CHP':      return chebyshevPolynomial(A, pDeg, al, be)
      elif prec == 'INV':      return Inverse(A)
      elif prec == 'MSPAI':    return MSPAI(A)
