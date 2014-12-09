import numpy as np
from ase.atoms import Atoms
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG
from winak.curvilinear.numeric.SparseMatrix import AmuB, svdB, eigB
from scipy import linalg as la

class Delocalizer:
    def __init__(self,atoms_obj,weighted=False,icList=None, periodic=False, dense=False):
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
        x0=self.x_ref.flatten()
        self.natoms=len(self.masses)
        #VCG constructs primitive internals (bond length,bend,torsion,oop)
        import gc
        gc.collect()
        if icList is None:
            if periodic:
                self.vcg=PVCG(atoms=self.atoms,masses=self.masses, cell=self.cell)
                self.iclist=self.vcg(x0)
            else:
                self.vcg=VCG(atoms=self.atoms,masses=self.masses)
                self.iclist=self.vcg(x0)
        else:
            self.iclist=icList
        if periodic:
            self.ic=Periodic_icSystem(self.iclist, len(self.atoms), 
                    masses=self.masses, xyz=x0, cell=self.cell)
        else:
            self.ic=icSystem(self.iclist,len(self.atoms),
                              masses=self.masses,xyz=x0)
        self.ic.backIteration = self.ic.denseBackIteration
        #no need to eval A
        #self.ic.evalA()
        self.ic.evalB()

        #generate 1/M matrix for mass weighed internals
        #use with care!
        if weighted:
            self.m=[]
            for i in range(self.natoms):
                mtemp=1/self.masses[i]
                for j in range(3):
                    mrow=np.zeros(self.natoms*3)
                    mrow[i*3+j]=mtemp
                    self.m.append(mrow.copy())
            self.m=np.asarray(self.m)
        else:
            self.m=np.identity(self.natoms*3)

        #thctk uses sparce matrices
        #b matrix is defined as delta(q)=b*delta(x)
        #where q are primitive internals and x cartesian coords
        if dense:
            self.b=self.ic.B.full()
            self.g=np.dot(self.b,self.m)
            self.g=np.dot(self.g,self.b.transpose())

            v2, self.ww,self.u=np.linalg.svd(self.g)
            self.u = self.u[:3*len(self.atoms_object)-6]
        else:
            if periodic:
                k = 3*len(self.atoms_object)
            else:
                k = 3*len(self.atoms_object)-6
            #Sparse algorithm
            self.b = self.ic.B
            bt = self.ic.evalBt(perm=0)
            self.g = AmuB(self.b,bt)
            v2, self.ww, self.u = svdB(self.g, k=k)
            self.u =self.u[:k]


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
