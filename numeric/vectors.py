import random
from thctk.numeric import *
from pysparse import spmatrix
import math
import copy
LA=importLinearAlgebra()

class BasisVectors:
  def __init__(self, a={}, factor=100.):
    self.CIM=a
    self.f=factor

  def randomVectorGen(self, vec, mode):
    V0=vec.copy()
    V2=[]
    for i in range(len(V0)):
      rd=random.random()*2-1
      V0[i]=rd/N.sqrt(len(V0))
      V2.append(V0[i]**2)
    Ve=N.array(V2)
    Ve/=self.f
    Ve[mode]=V0[mode]=1.0
    norm = N.sum(Ve)
    Ve/=norm
    V0=N.sqrt(Ve)
    return V0

  def gaussFunction(self, x, x0, amp, w, eps=1e-8):
    y = x - x0
    y /= (0.5*w + eps)
    y *= y*(-0.5)
    y = math.exp(y)
    return y

  def gaussVectorGen(self, mode, gw=0.001, amp=1):
    V0=N.zeros(self.CIM.shape[0])
    vec=V0.copy()
    tau=self.CIM[mode,mode]
    for i in range(len(V0)):
      rd=random.random()*2-1
      x=self.CIM[i,i]
      g=self.gaussFunction(x, tau, amp, gw)
      vec[i]=(rd*g)/self.f
      if i == mode:
        vec[i]=1.
    norm=N.sqrt((vec**2).sum())
    vec/=norm
    return vec

  def overlapMatrix(self, V):
    n=len(V)
    s=N.zeros((n, n), nxFloat)
    for i in range(n):
      s[i,i]=N.dot(V[i],V[i])
      for j in range(i+1,n):
        s[i,j]=s[j,i]=N.dot(V[i],V[j])
    return s

  def squareOverlapMatrix(self, sma):
    sm=N.array(sma)
    ev,u=LA.Heigenvectors(sm)
    tu=N.transpose(u)
    a=N.dot(u,sm)
    b=N.dot(a,u)
    v=N.zeros(N.shape(b), nxFloat)
    for i in range(len(b)):
      v[i,i]=N.sqrt(b[i,i])**(-1)
    c=N.dot(tu,v)
    sq=N.dot(c,tu)
    return sq

  def orthonormalVectors(self, sq, v):
    vo=N.dot(sq,v)
    return vo

  def calc(self, modes=[], gw=0.001):
    vec=[]
    for mode in modes:
      v=self.gaussVectorGen(mode, gw)
      vec.append(v)
    vc=N.array(vec)
    s=self.overlapMatrix(vc)
    sq=self.squareOverlapMatrix(s)
    vo=self.orthonormalVectors(sq,vc)
    return vo.T

class MatrixGenerator:
  def __init__(self, n, hint, bandwidth, sym=0):
    self.n = n
    self.hint = hint
    self.bandwidth = bandwidth
    if sym == 1:
      self.sym()
    elif sym == 0:
      self.asym()
    elif sym == 2:
      self.small()

  def small(self):
    n = self.n
    hint = self.hint
    bandwidth = self.bandwidth
    llm = spmatrix.ll_mat_sym(n,hint)
    for i in range(n):
      for j in range(i+1):
        if abs(i-j) < bandwidth:
          llm[i,j]=abs(random.random())
        if i == j:
          llm[i,j]=int(random.random()*10)+1
    self.matrix = llm   

  def sym(self):
    n = self.n
    hint = self.hint
    bandwidth = self.bandwidth
    llm = spmatrix.ll_mat_sym(n,hint)
    for i in range(n):
      for j in range(i+1):
        if abs(i-j) < bandwidth:
          llm[i,j]=int(random.random()*10)
        if llm[i,i] == 0:
          llm[i,i] += 1
    self.matrix = llm   

  def asym(self):
    n = self.n
    hint = self.hint
    bandwidth = self.bandwidth
    llm=spmatrix.ll_mat(n,hint)
    for i in range(n):
      for j in range(n):
        if abs(i-j) < bandwidth:
          llm[i,j]=int(random.random()*10)
        if llm[i,i] == 0:
          llm[i,i] += 1
    self.matrix = llm   

