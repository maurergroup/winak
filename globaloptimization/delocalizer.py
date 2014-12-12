import numpy as np
from ase.atoms import Atoms
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG
from winak.curvilinear.numeric.SparseMatrix import AmuB, svdB, eigB, CSR
from scipy.sparse import csr_matrix
from scipy import linalg as la

class Delocalizer:
    def __init__(self,atoms_obj,weighted=True,icList=None, periodic=False, 
            dense=False, normalized=False, threshold=0.5):
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
        self.normalized = normalized
        self.weighted = weighted
        x0=self.x_ref.flatten()
        self.natoms=len(self.masses)
        #VCG constructs primitive internals (bond length,bend,torsion,oop)
        if icList is None:
            if periodic:
                self.vcg=PVCG(atoms=self.atoms,masses=self.masses, cell=self.cell, \
                        threshold=threshold)
                self.iclist=self.vcg(x0)
            else:
                self.vcg=VCG(atoms=self.atoms,masses=self.masses, \
                        threshold=threshold)
                self.iclist=self.vcg(x0)
        else:
            self.iclist=icList

        self.initIC()
        self.evalG()
        if normalized:
            self.normalize_U()

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

    def evalG(self):
        dense = self.dense
        periodic = self.periodic
        weighted = self.weighted
        normalized = self.normalized

        if periodic:
            self.m=np.eye(3*self.natoms+9)
        else:
            self.m=np.eye(3*self.natoms)
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
            k = 3*self.natoms
        else:
            k = 3*self.natoms-6
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
            g=np.dot(b,self.m)
            g=np.dot(g,b.transpose())
            v2, self.ww,self.u=np.linalg.svd(g)
            self.uu =self.u.copy()
            self.u = self.u[:k]
            self.ww = self.ww[:k]
        else:
            #Sparse algorithm
            g = AmuB(self.ic.B,self.m)
            g = AmuB(g, self.ic.Bt)
            #g = AmuB(self.ic.B, self.ic.Bt)
            self.v2, self.ww, self.u = svdB(g, k=k)
            self.ww = self.ww[:k][::-1]
            self.uu =self.u.copy()
            self.u =self.u[:k][::-1]


    def get_U(self):
        return self.u

    def get_constrainedU(self):
        return self.u2

    def constrainStretches(self):
        str=self.ic.getStretchBendTorsOop()[0][0]#all stretches
        e=[]
        for i in str:
            d=np.zeros(len(self.ic))
            d[i]=1
            e.append(d)
        self.constrain(e)

    def normalize_U(self, v=None):
        #TESTING
        if v is None:
            u = self.uu
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

    def constrain(self,constraint):
        alright=True
        if isinstance(constraint, (list, tuple)):
            for i in constraint:
                if len(i)!=len(self.u[0]):
                    alright=False
        elif len(constraint)!=len(self.u[0]):
            alright=False
        else:
            temp=constraint
            constraint=[]
            constraint.append(temp)
        if alright:
            c=[]
            for i in range(len(constraint)):
                c.append(np.zeros(len(constraint[0])))
                for j in self.u:
                    c[i]+=np.dot(constraint[i],j)*j
            t=np.append(c,self.u)
            t=np.reshape(t,(len(self.u)+len(c),len(constraint[0])))
            self.vk=[]
            for i in range(len(t)):
                self.vk.append(t[i])
                for j in range(0,i):
                    self.vk[i]-=np.dot(t[i],self.vk[j])*self.vk[j]
            self.vkold=np.asarray(self.vk)

            self.vk=np.asarray(self.vk[len(c):len(self.vk)])

            u2t=self.vkold[:3*self.natoms-6]
            self.u2=u2t
