import numpy as np
from ase.atoms import Atoms
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG
from winak.curvilinear.numeric.SparseMatrix import AmuB, svdB, eigB, CSR
from scipy.sparse import csr_matrix, csc_matrix
from scipy import linalg as la
from scikits.sparse.cholmod import cholesky

class Delocalizer:
    def __init__(self,atoms_obj,weighted=True,icList=None, periodic=False, 
            dense=False, threshold=0.5, add_cartesians = True):
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
        self.dense = dense
        self.periodic = periodic
        self.weighted = weighted
        x0=self.x_ref.flatten()
        self.natoms=len(self.masses)
        self.add_cartesians = add_cartesians

        #VCG constructs primitive internals (bond length,bend,torsion,oop)
        if icList is None:
            if periodic:
                self.vcg=PVCG(atoms=self.atoms,masses=self.masses, cell=self.cell, \
                        threshold=threshold, add_cartesians=add_cartesians)
                self.iclist=self.vcg(x0)
            else:
                self.vcg=VCG(atoms=self.atoms,masses=self.masses, \
                        threshold=threshold, add_cartesians=add_cartesians)
                self.iclist=self.vcg(x0)
        else:
            self.iclist=icList

        self.initIC()
        self.evalG()

    def initIC(self):
        periodic = self.periodic
        dense = self.dense
        x0 = self.x_ref.flatten()
        if periodic:
            self.ic=Periodic_icSystem(self.iclist, len(self.atoms), 
                    masses=self.masses, xyz=x0, cell=self.cell)
        else:
            self.ic=icSystem(self.iclist,len(self.atoms),
                              masses=self.masses,xyz=x0)
        if dense:
            self.ic.backIteration = self.ic.denseBackIteration
        else:
            self.ic.backIteration = self.ic.sparseBackIteration

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
                #calculate approximate eigenvalues
                #if normalized:
                    #approx. prediag of G gives eigenvalues l with 
                    #which we scale BtXlXB
                
                #g = np.dot(b,b.transpose())
                #not_converged = True
                #eigs = np.diag(g)
                #i = 0
                #g = csc_matrix(g)
                #while not_converged:
                    #factor = cholesky(g)
                    #L = factor.L()
                    #g = L.transpose().dot(L)
                    #new_eigs = np.diag(g)
                    #conv = [np.abs(e-ne)<0.01 for e,ne in zip(eigs,new_eigs)]
                    #eigs = new_eigs.copy()
                    #i += 1
                    #if np.all(conv):
                        #not_converged=False
                #print eigs , i
                #raise SystemExit()

                mm = np.eye(self.n)

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
            import time
            print time.time()
            #Sparse algorithm
            if self.weighted:
                #do Baker
                g = AmuB(self.ic.B,self.m)
                g = AmuB(g, self.ic.Bt)
                print 'after AmuB ', time.time()
                self.ww, self.u = eigB(g, k=k)
                self.ww = np.real(self.ww[:k])
                self.u = np.real(self.u.transpose()[:k])
                print 'after SVD ', time.time()
            else:
                #do Andzelm
                b = self.ic.B
                bt = self.ic.Bt
                f = AmuB(bt, b)
                print 'after AmuB ', time.time()
                s2, self.ww, s = svdB(f, k=k)
                print 'after SVD ', time.time()
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
            #d=self.ic.B.full()[:,i]
            d[i]=1
            e.append(d)
        return e
    
    def constrainStretches2(self, stretch_list = None):
        str=self.ic.getStretchBendTorsOop()[0][0]#all stretches
        if stretch_list is None:
            stretch_list = str
        else:
            stretch_list = list(stretch_list)
        e=[]
        #we know stretches come first in the B matrix
        for i in stretch_list:
            d=self.ic.B.full()[:,i]
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
        for s in bend_ind[-3:]:
            d=np.zeros(len(self.ic))
            d[s]=1
            e.append(d)
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

        u = self.u.copy()
        b = self.ic.B.full().transpose()
        for i,row in enumerate(b):
            tmp = row
            for c in constraints:
                tmp -= (np.dot(tmp, c) * c) / np.dot(c,c)
            b[i] = tmp 
        #b = b[:,4:] 
        #self.ic.n = self.ic.n - 4
        #self.ic.ic = self.ic.ic[4*3:]
        #here we need to fix iclist or initialize a new ic system
        from scipy.sparse import csr_matrix
        print b.shape
        B = csr_matrix(b.transpose())
        B = CSR(n = B.shape[0], m = B.shape[1],
                nnz = B.nnz, i= B.indptr,
                j=B.indices, x= B.data)
        self.ic.Bnnz = B.nnz
        self.ic.B = B
        del self.ic.Bt
        self.ic.evalBt()
        self.evalG(c=0)
        print len(self.ic)
        self.u2 = self.u
        print self.u2.shape
        #print self.ic.Bnnz, self.ic.n
        #print self.ic()
        #print self.u.shape
        #raise SystemExit()

    def constrain(self,constraints):
        u = self.u
        #a number of constraint vectors
        #is projected onto the active coord. space
        n_constr = len(constraints)
        nn = len(u) - n_constr
        c = []
        #for coord in u:
            #tmp = coord.copy()
            #for con in constraints:
                #tmp -= np.dot(coord, con) * con
            #tmp /= np.linalg.norm(tmp)
            #c.append(tmp)

        #for con in constraints:
            #tmp = con
            #tmp = np.zeros_like(con)
            #for coord in u:
                #tmp += np.dot(coord, con) * coord
            ##tmp /= np.linalg.norm(tmp)
            #c.append(tmp)
            #print np.linalg.norm(c[0])

        constraints = np.array(constraints)
        constraints = np.dot(u.transpose(),np.dot(u, constraints.transpose())).transpose()
        c = list(constraints)
        for coord in u:
            c.append(coord)
        c[0] /= np.linalg.norm(c[0])
        u2 = [c[0]]
        for vec in c[1:]:
            tmp = vec
            for v2 in u2:
                tmp = tmp - (np.dot(tmp,v2)/np.dot(v2,v2)) * v2
            tmp /= np.linalg.norm(tmp)
            u2.append(tmp)
        self.u2 = np.array(u2[:nn])

