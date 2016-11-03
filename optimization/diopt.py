# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer

from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.globaloptimization.delocalizeer import *

class DIopt(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, npulaymax=15, ):
        """delocalized inernal optimizer following J. Chem. Phys. 122, 124508 (2005).
        written by Reinhard J. Maurer, Yale University 2016
        The optimizer has external dependencies, namely the curvilinear coordinate 
        code 'winak', see www.damaurer.at

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with 
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.
        
        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
 
        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.04 Å).
        
        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        if maxstep is not None:
            if maxstep > 1.0:
                raise ValueError('You are using a much too large value for ' +
                                 'the maximum step size: %.1f Å' % maxstep)
            self.maxstep = maxstep

        self.npulaymax = 15
        
        self.k = [0.45, 0.15, 0.005]
        self.alpha = np.array([
            [1.000,0.3949,0.3949],
            [0.3949,0.2800,0.2800],
            [0.3949,0.2800,0.2800],
            ])
        self.r0 = np.array([
            [1.35,2.10,2.53],
            [2.10,2.87,3.40],
            [2.53,3.40,3.40],
            ])
        elements_dict = {}
        for i in range(1,119):
            if i<3:
                elements_dict[i] = 0
            elif i<11:
                elements_dict[i] = 1
            else:
                elements_dict[i] = 2
        self.el_dict

    def initialize(self):
        #initialize curvilinear coordinates
        self.is_periodic = any(self.atoms.get_pbc())
        d = Delocalizer(atoms,periodic=self.is_periodic,dense=True, weighted=True)

        if self.is_periodic:
            pass
        else:
            self.coords = DC(d.x_ref.flatten(), d.masses, internal=True, 
                    atoms=d.atoms, ic=d.ic, L=None, Li=None, u=d.get_U(), 
                    biArgs={'RIIS': 4, 'RIIS_maxLength': 6, 'maxStep':0.5,})


        #initialize all the memory stuff for DIIS
        self.r_diis = []
        self.f_diis = []

        #setup model Hessian
        self.H = self.get_hessian()

    def read(self):
        self.H, self.r0, self.f0, self.maxstep = self.load()

    def step(self, f):
        c = self.coords
        atoms = self.atoms
        r = atoms.get_positions().flatten()
        f = f.flatten()
        #stress
        if self.is_periodic:
            #do stress
            s = self.atoms.get_stress().flatten()
            # pass

        #transform to internal coordinates
        if is_periodic:
            sr = c.getS(np.concatenate([r,atoms.cell.flatten()]))
            sf = c.grad_x2s(np.concatenate([f,s]))
        else:
            sr = c.getS(r)
            sf = c.grad_x2s(f)

        #make step in DCs
        ds = self.update(sr, sf)
        
        #transform displacement to cartesian
        #TODO Coordinates objects should have a function that 
        #applies B, sparse
        b = c.evalBvib()
        dr = np.dot(b,ds)

        atoms.set_positions(r + dr)
        f = atoms.get_forces() 
        return f

    def update(self, r, f, r0, f0):
        """
        coordinate step in DC coordinates
        """
      
        #update Hessian matrix with BFGS
            #get new hessian
            #update old hessian

        #do quasi-Newton for the first few steps
        dr = np.dot(np.inv(self.H),f)

        
        #riis stuff

        # self.r_diis.append(r)
        # self.f_diis.append(f)
        #given the new forces and the old ones 
        #in self.f_diis we build a matrix
        # dr, w = doRIIS(self.r_diis,self.f_diis,dim=len(self.r_diis))



        return dr

    def get_hessian(self):
        """
        model hessian based on Bucko et al.
        this function assumes that self.coords and atoms.positions
        are synchronized and up to date.
        """
        c = self.coords
        U = c.Li
        Ut = c.L
        ns =len(c.ic)

        Hdiag = np.zeros([ns])
        ic = c.ic.ic
        ind = 0
        a = 0
        xyz = ic.xyz
        while ind<len(ic):
            if ic[ind] == 0:
                break
            elif ic[ind] == 1:
                #bond
                i = ic[ind+1]-1 
                j = ic[ind+2]-1
                rho1 = self.get_exp(i,j)
                Hdiag[a] = self.k[0]*rho1
                ind += 3
            elif ic[ind] == 2:
                #angle
                i = ic[ind+1]
                j = ic[ind+2]
                k = ic[ind+3]
                rho1 = self.get_exp(i,j)
                rho2 = self.get_exp(j,k)
                Hdiag[a] = self.k[1]*rho1*rho2 
                ind += 4
            elif ic[ind] == 3:
                #dihedral
                i = ic[ind+1]
                j = ic[ind+2] 
                k = ic[ind+3] 
                l = ic[ind+4]
                rho1 = self.get_exp(i,j)
                rho2 = self.get_exp(j,k)
                rho3 = self.get_exp(k,l)
                Hdiag[a] = self.k[2]*rho1*\
                        rho2*rho3
                ind += 5
            else:
                break
            a += 1

        return np.dot(Ut,np.dot(np.diag(Hdiag),U)
           
    def get_exp(self,i, j):
        pos = self.atoms.positions
        elnums = self.atoms.get_atomic_numbers()
        rij = (pos[i]-pos[j])
        r0ij = self.r0[self.el_dict[elnums[i]],self.el_dict[elnums[j]]]
        alpha = self.alpha[self.el_dict[elnums[i]],self.el_dict[elnums[j]]]
        return np.exp(alpha(r0ij*r0ij - rij*rij))


    # def doRIIS(x, e, dim = 3):
        # """
        # Do an Regularized Inversion of the Iterative Subspace (RIIS) for each atom.
        # 'dim' gives an approximate cutoff for the Tikhonov regularization.
        # HINT: This could be optimized by saving the correlation matrices M for 
        # each atom and calculating only the new vector after each iteration step.
        # """
        # assert len(e) == len(x)
        # # order the error vectors and positions of the atoms so, that there is
        # # a list of those vectors for each atom.
        # X = N.rollaxis(N.asarray(x), 1,0)
        # E = N.rollaxis(N.asarray(e), 1,0)
        # Xn = N.empty(x[0].shape) # the new interpolated geometry
        # n = len(e) + 1
        # M = -N.ones((n,n) ) # correlation matrix
        # b = N.zeros(n) # inhomogeneity of the linear equation system
        # w = N.zeros(n) # interpolation weights
        # M[-1,-1] = 0
        # b[-1] = -1
        # for (x, e, xn) in zip(X, E, Xn):
            # for i in xrange(len(e)):
                # for j in range(i+1):
                    # M[i,j] = M[j,i] = N.dot(e[i], e[j])
            # # do Tikhonov regularization
            # U, s, VT = N.linalg.svd(M)
            # eps = s[dim] # eps is determined by the singular values 
            # s2 = s*s 
            # s2 += eps*eps
            # s /= s2
            # Mi = N.dot(U, s[:,N.newaxis]*VT) # regularized inverse
            # w = N.dot(Mi, b) # approximate weights
            # xn[:] = sum( w_i*x_i for (w_i, x_i) in zip(w[:-1], x))
            # return Xn.flatten(), w[:-1]
