from thctk.numeric import *     
LA = importLinearAlgebra()

class NSOLD:

  """
  dense Newton solver for systems of nonlinear equations

  Ref.: C.T. Kelley, "Solving Nonlinear Equations with 
  Newton's Method"

  derived from Matlab code nsold.m
  """

  def __init__(self, function, atol=0.01, rtol=0.01, rsham=0.5, maxarm=20, 
      maxit=40, epsdirder = 1.0e-7, debug=0, norm = None):

    if norm is None:
      self.norm  = lambda x: N.sqrt(N.dot(x, x))
    else:
      self.norm = norm
    self.function = function
    self.atol, self.rtol = atol, rtol
    self.maxit, self.maxarm = maxit, maxarm
    self.rsham = rsham
    self.outstat = []
    self.debug = debug
    self.epsdirder = epsdirder
 
  def __call__(self, x):
    try:
      self.dim = len(x)
      diffjac = self.diffjacN
      self.solve = LA.solve_linear_equations
    except:
      self.dim = 1
      diffjac = self.diffjac1
      self.solve = lambda x,y: y / x
    try:
      analyticJacobian = self.function.jacobi
      diffjac = lambda x, f: analyticJacobian(x)
    except AttributeError: # no analytic Jacobian available
      pass
    f0 = self.function(x) 
    fnrm = self.norm(f0)
    stop_tol = self.atol + self.rtol * fnrm
    it_hist = [[fnrm, 0]]
    itc = 0
    armflag = 0
    fnrmo = 1
    while fnrm > stop_tol and itc < self.maxit:
      rat = fnrm / fnrmo
      self.outstat.append((itc, fnrm, rat))
      fnrmo = fnrm
      itc += 1

      if itc == 1 or rat > self.rsham or armflag == 1:
        jac_age = -1
        jac = diffjac(x, f0)
     
      direction = -self.solve(jac, f0)
      jac_age += 1
      xold, fold  = x, f0
      step, iarm, x, f0, armflag = self.armijo(direction, x, f0)
#     print step, iarm, x 
      if armflag == 1: 
         if jac_age > 0:
            sol = xold
            x, f0 = xold, fold
            print 'Armijo failure; recompute Jacobian.'
         else:
            print 'Complete Armijo failure.'
            x, ierr = xold, 2
            break
      fnrm = self.norm(f0)
      it_hist.append([fnrm,iarm])
      rat = fnrm / fnrmo
      if self.debug == 1: print itc, fnrm, rat, x
      self.outstat.append((itc, fnrm, rat))
    sol = x
    if self.debug == 1: print self.outstat
    if fnrm > stop_tol: ierr = 1
    else: ierr = 0 
    return sol, it_hist, ierr
  

  def diffjac1(self, x, f0):
    epsnew = self.epsdirder
    if x > 0:     epsnew *= max(x, 1)
    elif x < 0:  epsnew *= -max(abs(x), 1)
    f1 = self.function(x + epsnew)
    return (f1 - f0)/epsnew

  def diffjacN(self, x, f0):
    n = self.dim
    jac = N.zeros((n,n), nxFloat)
    for j in range(n):
      zz = N.zeros(n)
      zz[j] = 1
      jac[j] = self.dirder(x, zz, f0)
   #print jac
    return jac

  def dirder(self, x, w, f0):
    n = self.dim
    epsnew = self.epsdirder
    if self.norm(w) == 0:
      return N.zeros(n)
    xs = N.dot(x, w) / self.norm(w)
    if xs > 0:
      epsnew *= max(xs, 1)
    elif xs < 0:
      epsnew *= -max(abs(xs), 1)
    epsnew /= self.norm(w)
    delta = x + epsnew * w
    f1 = self.function(delta)
    z = (f1 - f0) / epsnew
    return z

  def armijo(self, direction, x, f0, alpha=1.e-4, sigma1=.5, lambd=1., lamm=1.): 
    maxarm = self.maxarm
    xold = x
    lamc = lambd 
    step = lambd * direction
    xt = x + step
    ft = self.function(xt)
    nft = self.norm(ft)
    nf0 = self.norm(f0)
    ff0 = nf0**2 
    ffc = nft**2
    ffm = nft**2
    iarm = 0
    while nft >= (1 - alpha * lambd) * nf0:
      if iarm == 0:
        lambd *= sigma1
      else:
        lambd = self.parab3p(lamc, lamm, ff0, ffc, ffm)
      step = lambd * direction
      xt = x + step
      lamm, lamc = lamc, lambd
      ft = self.function(xt)
      nft = self.norm(ft)
      ffm, ffc = ffc, nft**2
      iarm += 1
      if iarm > maxarm:
        print 'Armijo failure, too many reductions'
        armflag = 1
        sol = xold
        return step, iarm, xt, ft, armflag
    return step, iarm, xt, ft, 0

  def parab3p(self, lambdac, lambdam, ff0, ffc, ffm, sigma0=.1, sigma1=.5):
    c2 = lambdam * (ffc - ff0) - lambdac * (ffm - ff0)
    if c2 >= 0: return sigma1 * lambdac
    c1 = lambdac * lambdac * (ffm - ff0) - lambdam * lambdam * (ffc - ff0)
    lambdap = -c1 * 0.5 / c2
    if   lambdap < sigma0 * lambdac: return sigma0 * lambdac
    elif lambdap > sigma1 * lambdac: return sigma1 * lambdac
    else: return lambdap

class simpleFunction:
  """
  2D example function with analytical Jacobian
  """

  def __init__(self): pass

  def __call__(self, key):
    x1, x2 = key
    f1 = x1**2 + x2**2-2
    f2 = N.exp(x1-1) + x2**2 - 2
    F = N.array([f1,f2], nxFloat)
    return F

  def jacobi(self, x):
    x1, x2 = x
    return N.array([
    [ 2*x1, 2*x2 ],
    [ N.exp(x1 -1 ), 2*x2 ]
    ])

