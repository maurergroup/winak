import numpy as np
from ase.atoms import Atoms
from INTERNALS.curvilinear.InternalCoordinates import icSystem
from INTERNALS.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from INTERNALS.curvilinear.Coordinates import InternalCoordinates as IC
#from thctk.constants import UNIT
from scipy import linalg as la

class Delocalizer:
    def __init__(self,atoms_obj,weighted=False,icList=None):
        """This generates the delocalized internals as described in the
        paper.
        atoms_obj: a properly initialized ase Atoms object (positions,
        masses, symbols)
        weighted: boolean, if you want mass weighted delocalized internals
        icList: if you want to supply your own icList, simply hand it over
        """
        self.atoms_object = atoms_obj
        self.x_ref=atoms_obj.get_positions()
        self.masses  = atoms_obj.get_masses()
        self.atoms = atoms_obj.get_chemical_symbols()
        self.vk=None
        self.vc=None
        x0=self.x_ref.flatten()
        #VCG constructs primitive internals (bond length,bend,torsion,oop)
        if icList is None:
            self.vcg=VCG(self.atoms,masses=self.masses)
            self.iclist=self.vcg(x0)
        else:
            self.iclist=icList
        self.ic=icSystem(self.iclist,len(self.atoms),
                          masses=self.masses,xyz=x0.copy())
        self.ic.backIteration = self.ic.denseBackIteration
        self.ic.evalA()
        self.ic.evalB()

        self.natoms=len(self.masses)

        #generate 1/M matrix for mass weighed internals
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
        self.b=self.ic.B.full()
        self.g=np.dot(self.b,self.m)
        self.g=np.dot(self.g,self.b.transpose())

        self.v2,self.ww,self.u=np.linalg.svd(self.g)
        self.u = self.u[:3*len(self.atoms_object)-6]
        self.bnew=np.dot(self.u,self.b) #b'=u*b
        self.g=np.dot(self.bnew,self.bnew.transpose())
        ww,self.v=la.eig(self.g)
        self.ginv=np.dot(self.v,np.dot(np.eye(len(self.g))*(1./ww),
                         self.v.transpose()))
        self.left=np.dot(self.bnew.transpose(),self.ginv.transpose())

        self.w=[]

        for i in range(len(self.left.transpose())):
            self.w.append(self.left.transpose()[i].reshape(self.natoms,3))
            self.w[i]/=np.max(np.abs(self.w[i]))
        self.w=np.asarray(self.w)



    def get_vectors(self):
        """Returns the delocalized internal eigenvectors as cartesian
        displacements. Careful! get_vectors()[0] is the first vector.
        If you want to interpret it as a matrix in the same way numpy does,
        you will have to transpose it first."""
        return self.w

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
            self.vc=[]

            #normales left KANN NICHT GEHEN!
            #for i in range(len(self.left.transpose())):
            #    self.vc.append(np.dot(self.left.transpose(),self.vk[i]))
            #    self.vc[i]/=np.max(np.abs(self.w[i]))
            #self.vc=np.asarray(self.vc)
            #self.vc=np.reshape(self.vc,(12,6,3))

            #neues B Welle, not working great
            u2t=self.vkold[:3*self.natoms-6]
            self.u2=u2t#.transpose()
            self.b2=np.dot(self.u2,self.b)
            self.g2=np.dot(self.b2,self.b2.transpose())
            ww,self.v2=la.eig(self.g2)
            self.ginv2=np.dot(self.v2,np.dot(np.eye(len(self.g2))*(1./ww),self.v2.transpose()))
            self.le=np.dot(self.b2.transpose(),self.ginv2.transpose())
            self.vc=[]
            for i in range(len(self.le.transpose())):
                self.vc.append(self.le.transpose()[i].reshape(self.natoms,3))
            self.vc=np.asarray(self.vc)
            self.vc=np.reshape(self.vc,(len(self.le.transpose()),self.natoms,3))

            #left von B alt
            #self.ge=np.dot(self.b,self.b.transpose())
            #self.le=np.dot(np.linalg.inv(self.ge),self.b)
            #for i in range(len(self.vk)):
            #    self.vc.append(np.dot(self.le.transpose(),self.vk[i]))
            #self.vc=np.asarray(self.vc)
            #self.vc=np.reshape(self.vc,(12,6,3))


            for i in range(len(self.vc)):
                self.vc[i]/=np.max(np.abs(self.vc[i]))
                #self.vc[i]*=1.5


    def get_constrainedvectors(self):
        """Returns the constrained eigenvectors as Cartesian displacements."""
        return self.vc

    def get_constraineddelocvectors(self):
        """Returns the constrained delocalized eigenvectors."""
        return self.vk

    def get_delocvectors(self):
        """Returns the delocalized internal eigenvectors."""
        return self.u.transpose()

    def write_jmol(self,filename,constr=False):
        """This works similar to write_jmol in ase.vibrations."""
        fd = open(filename, 'w')
        if constr:
            wtemp=self.vc
        else:
            wtemp=self.w
        for i in range(len(self.left.transpose())):
            fd.write('%6d\n' % self.natoms)
            fd.write('Mode #%d, f = %.1f%s cm^-1 \n' % (i, self.ww[i], ' '))
            for j, pos in enumerate(self.atoms_object.positions):
                fd.write('%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f \n' %
                     (self.atoms[j], pos[0], pos[1], pos[2],
                      wtemp[i,j, 0], wtemp[i,j, 1], wtemp[i,j, 2]))
        fd.close()
