# winak.globaloptimization.delocalizer
#
#    winak - python package for structure search and more in curvilinear coordinates
#    Copyright (C) 2016  Reinhard J. Maurer and Konstantin Krautgasser 
#    
#    This file is part of winak 
#        
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>#

import numpy as np
from ase.atoms import Atoms
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import Periodic_icSystem2
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG
from winak.curvilinear.numeric.SparseMatrix import AmuB, svdB, eigB, CSR
from scipy.sparse import csr_matrix, csc_matrix
from scipy import linalg as la
from scikits.sparse.cholmod import cholesky

class Delocalizer:
    def __init__(self,atoms_obj,icList=None, u=None, weighted=False, periodic=False,
            dense=False, threshold=0.5, add_cartesians = False,
            manual_bonds = [], expand=1,
            bendThreshold=170, torsionThreshold=160, oopThreshold=30):
        """This generates the delocalized internals as described in the
        paper.
        atoms_obj: a properly initialized ase Atoms object (positions,
        masses, symbols)
        weighted: boolean, if you want mass weighted delocalized internals
        icList: if you want to supply your own icList, simply hand it over (iclist oder Leben!)
        """
        self.atoms_object = atoms_obj
        self.x_ref=atoms_obj.positions
        self.masses  = atoms_obj.get_masses()
        self.atoms = atoms_obj.get_chemical_symbols()
        self.cell = atoms_obj.get_cell()
        self.vk=None
        self.vc=None
        self.u2=None
        self.expand=expand
        self.dense = dense
        self.periodic = periodic
        self.weighted = weighted
        x0=self.x_ref.flatten()
        self.natoms=len(self.masses)
        self.add_cartesians = add_cartesians
        self.constraints_int = []
        self.constraints_cart = []
        #VCG constructs primitive internals (bond length,bend,torsion,oop)
        if icList is None:
            if periodic:
                self.vcg=PVCG(atoms=self.atoms,masses=self.masses, cell=self.cell, \
                        threshold=threshold, add_cartesians=add_cartesians, 
                        manual_bonds = manual_bonds, expand=self.expand)
                self.iclist=self.vcg(x0, bendThreshold=bendThreshold,
                        torsionThreshold=torsionThreshold, oopThreshold=oopThreshold)
            else:
                self.vcg=VCG(atoms=self.atoms,masses=self.masses, \
                        threshold=threshold, add_cartesians=add_cartesians,
                        manual_bonds = manual_bonds)
                self.iclist=self.vcg(x0, bendThreshold=bendThreshold,
                        torsionThreshold=torsionThreshold, oopThreshold=oopThreshold)
        else:
            self.iclist=icList

        self.initIC()

        if u is None:
            self.evalG()
        else:
            self.u = u

    def initIC(self):
        periodic = self.periodic
        dense = self.dense
        x0 = self.x_ref.flatten()
        if periodic:
            self.ic=Periodic_icSystem2(self.iclist, len(self.atoms),
                    masses=self.masses, xyz=x0, cell=self.cell, expand=self.expand)
        else:
            self.ic=icSystem(self.iclist,len(self.atoms),
                              masses=self.masses,xyz=x0)
        if dense:
            self.ic.backIteration = self.ic.denseBackIteration
            # self.ic.ig = self.ic.denseinternalGradient
        else:
            self.ic.backIteration = self.ic.sparseBackIteration
            # self.ic.ig = self.ic.denseinternalGradient
            # self.ic.ig = self.ic.sparseinternalGradient

    def evalG(self, c=0):
        dense = self.dense
        periodic = self.periodic
        weighted = self.weighted

        if periodic:
            self.m=np.eye(3*self.natoms+9-c)
        else:
            self.m=np.eye(3*self.natoms-c)
        #generate 1/M matrix for mass weighed internals
        #use with care!
        if weighted:
            for i in range(self.natoms):
                mtemp=1./self.masses[i]
                for j in range(3):
                    self.m[i*3+j,i*3+j]=mtemp
            if periodic:
                mtemp = self.masses.sum()
                for i in range(1,10):
                    self.m[-i,-i]=1./mtemp
        if periodic:
            k = 3*self.natoms - c
            if self.add_cartesians:
                k += 6
        else:
            k = 3*self.natoms-6 - c
        if dense:
            pass
        else:
            M = self.m
            self.m = csr_matrix(self.m)
            self.m = CSR(n = self.m.shape[0], m = self.m.shape[0],
                    nnz = self.m.nnz, i= self.m.indptr,
                    j=self.m.indices, x= self.m.data)
        #b matrix is defined as delta(q)=b*delta(x)
        #where q are primitive internals and x cartesian coords
        if dense:
            b=self.ic.B.full()
            
            #Baker
            if self.weighted:
                g=np.dot(b,self.m)
                g=np.dot(g,b.transpose())
                v2, self.ww,self.u=np.linalg.svd(g)
                self.u = self.u[:k][::-1]
                self.ww = self.ww[:k][::-1]
            else:
                #g = np.dot(b.transpose(),b)
                #not_converged = True
                #eigs = np.diag(g)
                #i = 0
                #g = csc_matrix(g)
                #while not_converged:
                    #factor = cholesky(g,0.00001,mode='supernodal')
                    #L = factor.L()
                    #g = L.conjugate().transpose().dot(L)
                    #new_eigs = np.diag(g.todense())
                    #conv = [np.abs(e-ne)<0.1 for e,ne in zip(eigs,new_eigs)]
                    #eigs = new_eigs.copy()
                    #i += 1
                    #if np.all(conv):
                        #not_converged=False
                #print eigs , i

                mm = np.eye(self.ic.n)
                #mm[:len(eigs),:len(eigs)] = np.diag(1.0/eigs)
                #Andzelm
                f = np.dot(b.transpose(), mm)
                f = np.dot(f, b)
                
                s2, self.ww, s = np.linalg.svd(f)
                ww = 1./(np.sqrt(self.ww[:k]))
                self.ww = self.ww[:k][::-1]
                ww = np.diag(ww)
                s = (s[:k]).transpose()
                self.u = np.dot(b, np.dot(s, ww)).transpose()[::-1]
        else:
            #Sparse algorithm
            if self.weighted:
                #do Baker
                g = AmuB(self.ic.B,self.m)
                g = AmuB(g, self.ic.Bt)
                self.ww, self.u = eigB(g, k=k)
                self.ww = np.real(self.ww[:k])
                self.u = np.real(self.u.transpose()[:k])
            else:
                #do Andzelm
                b = self.ic.B
                bt = self.ic.Bt
                f = AmuB(bt, b)
                s2, self.ww, s = svdB(f, k=k)
                ww = 1./(np.sqrt(self.ww[:k]))
                ww = np.diag(ww)
                s = (s[:k]).transpose()
                ss = np.dot(s,ww)
                self.u = np.dot(b.full(), ss).transpose()

    def get_U(self):
        return self.u

    def get_constrained_U(self):
        return self.u2

    def constrainStretches(self, stretch_list = None):
        str=self.ic.getStretchBendTorsOop()[0][0]#all stretches
        if stretch_list is None:
            stretch_list = str
        else:
            stretch_list = list(stretch_list)
        e=[]
        #we know stretches come first in the B matrix
        for i in stretch_list:
            d=np.zeros(len(self.ic))
            d[i]=1
            self.constraints_int.append(i)
            e.append(d)
        return e
    
    def constrainCell(self):
        #the last 3 stretches and the last three bends are special 
        #cell DoFs
        e = []
        na = self.natoms
        SBTOC = self.ic.getStretchBendTorsOopCart()
        stretch_ind, bend_ind = SBTOC[0][0],SBTOC[0][1]
        for s in stretch_ind[-3:]:
            d=np.zeros(len(self.ic))
            d[s]=1
            e.append(d)
            self.constraints_int.append(s)
        for s in bend_ind[-3:]:
            d=np.zeros(len(self.ic))
            d[s]=1
            e.append(d)
            self.constraints_int.append(s)
        tmp = range(self.natoms*3,self.natoms*3+9)
        for i in tmp:
            self.constraints_cart.append(i)
        return e

    def constrainAtoms(self, atom_list):
        atom_list = list(atom_list)
        SBTOC = self.ic.getStretchBendTorsOopCart()[0]
        sbto_sum = 0
        for i in range(4):
            sbto_sum += len(SBTOC[i])
        e = []
        for c, xyz in atom_list:
            d = np.zeros(len(self.ic))
            d[sbto_sum+xyz*self.natoms+c] = 1
            self.constraints_cart.append(3*c+xyz)
            #self.constraints_int.append(3*c+xyz)
            e.append(d)
        return e

    def normalize_U(self, v=None):
        #TESTING
        if v is None:
            u = self.u
        else:
            u = v
        mu = np.dot(u, u.transpose())
        vv, ee, uu = np.linalg.svd(mu)
        if self.periodic:
            self.u = uu[:3*self.natoms] 
        else:
            self.u = uu[:3*self.natoms-6]
        #for i, vec in enumerate(self.u):
            #norm = np.linalg.norm(vec)*np.sqrt(self.ww[i])
            #u[i,:] /= norm

    def constrain2(self,constraints):
        ###TESTING ####
        
        def null(A, eps=1e-15):
            import scipy 
            u, s, vh = la.svd(A)
            null_mask = (s <= eps)
            null_space = scipy.compress(null_mask, vh, axis=0)
            return scipy.transpose(null_space)
        
        #constrained eigenvalue problem
        dofs = len(self.u)
        b = self.ic.B.full()
        n = self.ic.n
        c = np.array(constraints)
        m = len(c)
        c = c.transpose()
        #calculating nullspace
        c_tmp = np.zeros([n,n])
        c_tmp[:,:m] = c
        z = null(c_tmp)
        print np.dot(c_tmp,z)
        #solving generalized eigenvalue problem 
        zb = np.dot(z.transpose(),b)
        S = np.dot(z.transpose(),z)
        gg = np.dot(zb, zb.transpose())
        #from scipy.sparse.linalg import eigs
        E, V = la.eig(gg,S)#k=dofs, M=S)
        #vv,E, V = la.svd(gg,S)#k=dofs, M=S)
        #E, V = eigs(gg,k=dofs-m,M=S)#k=dofs, M=S)
        print E
        print V
        self.z = z
        #print np.dot(z,V)
        #print c.shape, self.u.shape
        #print np.dot(self.u,c)
        self.V = np.real(V.transpose())
        self.u2 = np.real(np.dot(z,V).transpose())
        print np.dot(self.u2,c)
        raise SystemExit()

    def constrain(self,constraints):
        u = self.u
        #a number of constraint vectors
        #is projected onto the active coord. space
        n_constr = len(constraints)
        nn = len(u)
        c = []
        for con in constraints:
            tmp = con
            tmp = np.zeros_like(con)
            for coord in u:
                tmp += np.dot(coord, con) * coord
            tmp /= np.linalg.norm(tmp)
            c.append(tmp)
        cc = c    
        #(aa,bb) = np.linalg.qr(np.array(c).transpose(),mode='full')
        #cc = list(aa.transpose()) 
       
        for coord in u:
            cc.append(coord)
        cc[0] /= np.linalg.norm(cc[0])
        u2 = [cc[0]]
        for vec in cc[1:]:
            tmp = vec
            for v2 in u2:
                tmp = tmp - (np.dot(tmp,v2)/np.dot(v2,v2)) * v2
            tmp /= np.linalg.norm(tmp)
            u2.append(tmp)
        self.u2 = np.array(u2[:len(u)])

