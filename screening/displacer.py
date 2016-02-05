from abc import ABCMeta, abstractmethod
from ase.all import *
from ase.utils.geometry import sort
import numpy as np
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.InternalCoordinates import icSystem
from winak.globaloptimization.disspotter import DisSpotter
from winak.screening.composition import *
import os

class Displacer:
    """This class accepts or declines a step in any way you see fit. It is also 
    in charge of logging."""
    __metaclass__ = ABCMeta

    def __init__(self):
        """subclasses must call this method."""
        pass
        
    @abstractmethod
    def displace(self,tmp):
        """subclasses must implement this method. Has to return a displace 
        ase.atoms object"""
        pass
    
    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing
        all the important parameters"""
        pass
    

class MultiDI(Displacer):
    def __init__(self,stepwidth,numdelocmodes=1,constrain=False,adsorbate=None,cell_scale=[1.0,1.0,1.0],adjust_cm=True,periodic=False,loghax=False):
        """cell_scale: for translations; scales translations, so everything
        stays in the unit cell. The z component should be set to something small,
        like 0.05. 
        numdelocmodes: how many delocalized internals should be combined; from
        0 to <1 it will interpret it as a percentage (i.e. 0.25=25%), 1 or any
        number larger than 1 will be interpreted as a certain number of modes 
        (i.e. 1=1 deloc mode, 15=15 delocmodes etc.)"""
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
        else:
            self.adsorbate=adsorbate
            self.ads=True
        self.cell_scale=cell_scale
        self.constrain=constrain
        self.adjust_cm=adjust_cm
        self.stepwidth=stepwidth
        self.numdelmodes=np.abs(numdelocmodes)
        self.lh=loghax
        self.periodic=periodic
        
    def displace(self,tmp):
        atms=tmp.copy()
        if self.ads:
            ads=tmp[self.adsorbate[0]:self.adsorbate[1]].copy()
            tmp1=tmp[:self.adsorbate[0]].copy()
            tmp2=tmp[self.adsorbate[1]:].copy()
            if len(tmp1)==0:
                surf=tmp2
            elif len(tmp2)==0:
                surf=tmp1
            else:
                surf=tmp1
                for i in tmp2:
                    surf.append(i)
        else:
            ads=tmp.copy()
        d=DisSpotter(ads)
        a=d.get_fragments()
        #tmp.write('overall.xyz')
        o=1
        if self.lh:
            while True:
                try:
                    os.stat(str(o))
                    o+=1
                except:
                    os.mkdir(str(o))       
                    break
            tmp.write(os.path.join(str(o),'pre_overall.xyz'))
        for i in a:
            i=np.asarray(i)-1
            if self.ads:
                tt=surf.copy()
                for j in ads[i]:
                    tt.append(j)
                adstmp=[len(surf),len(tt)]
                ads0=len(surf)
            else:
                tt=tmp[i]
                adstmp=None
                ads0=0
            
            di=DI(self.stepwidth,numdelocmodes=self.numdelmodes,constrain=self.constrain,adsorbate=adstmp,cell_scale=self.cell_scale,adjust_cm=self.adjust_cm,periodic=self.periodic)
            if self.lh:
                tt.write(os.path.join(str(o),'pre_'+str(i[0])+'.xyz'))
            tt=di.displace(tt)
            if self.lh:
                tt.write(os.path.join(str(o),'post_'+str(i[0])+'.xyz'))
            #tt.write(str(i[0])+'.xyz')
            k=0
            for j in i:
                atms[ads0+j].set('position',tt[ads0+k].get('position'))
                k+=1
        if self.lh:
            atms.write(os.path.join(str(o),'post_overall.xyz'))
        return atms
    
    def print_params(self):
        if self.ads:
            ads=', %d:%d is displaced',(self.adsorbate[0],self.adsorbate[1])
        else:
            ads=''
        if self.constrain:
            cc=''
        else:
            cc=' not'
        return '%s: stepwidth=%f%s, stretches are%s constrained'%(self.__class__.__name__,self.stepwidth,ads,cc)
        
class DI(Displacer):
    def __init__(self,stepwidth,numdelocmodes=1,constrain=False,adsorbate=None,cell_scale=[1.0,1.0,1.0],adjust_cm=True,periodic=False):
        """cell_scale: for translations; scales translations, so everything
        stays in the unit cell. The z component should be set to something small,
        like 0.05. 
        numdelocmodes: how many delocalized internals should be combined; from
        0 to <1 it will interpret it as a percentage (i.e. 0.25=25%), 1 or any
        number larger than 1 will be interpreted as a certain number of modes 
        (i.e. 1=1 deloc mode, 15=15 delocmodes etc.)"""
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
        else:
            self.adsorbate=adsorbate
            self.ads=True
            self.cell_scale=cell_scale
        self.constrain=constrain
        self.adjust_cm=adjust_cm
        self.stepwidth=stepwidth
        self.numdelmodes=np.abs(numdelocmodes)
        self.periodic=periodic
        
    def get_vectors(self,atoms):
        if self.periodic:
            deloc=Delocalizer(atoms,periodic=True,dense=True)
            coords=PC(deloc.x_ref.flatten(), deloc.masses, atoms=deloc.atoms, ic=deloc.ic, Li=deloc.u)
        else:
            deloc=Delocalizer(atoms)
            tu=deloc.u
            if self.constrain:
                e = deloc.constrainStretches()
                deloc.constrain(e)
                tu=deloc.u2
            if not self.ads:
                coords=DC(deloc.x_ref.flatten(), deloc.masses, atoms=deloc.atoms, ic=deloc.ic, u=tu)
            else:
                cell = atoms.get_cell()*self.cell_scale
                coords=CDC(deloc.x_ref.flatten(), deloc.masses, unit=1.0, atoms=deloc.atoms, ic=deloc.ic, u=tu,cell=cell)

        return coords.get_vectors() 
    
    def displace(self,tmp):
        """No DIS for atoms! Atoms get Cart movements. If on surf, along cell"""
        disp=np.zeros((len(tmp),3))
        ro=tmp.get_positions()
        atms=tmp.copy()
        if self.ads:
            tmp=tmp[self.adsorbate[0]:self.adsorbate[1]].copy()
            ads1=self.adsorbate[0]
            ads2=self.adsorbate[1]
        else:
            ads1=0
            ads2=len(tmp)
        if len(tmp)>1:
            if self.numdelmodes<1:
                if self.ads:
                    nummodes=int(np.round(self.numdelmodes*len(tmp)*3))#3N dis in adsorbates
                else:
                    nummodes=int(np.round(self.numdelmodes*(len(tmp)*3-6)))#3N-6 dis in gas phase
            else:
                nummodes=int(np.round(self.numdelmodes))#round and int just for you trolls out there 
            
            vectors=self.get_vectors(tmp)
            numvec=len(vectors)
            start=0
            if self.constrain:
                mm=tmp.get_masses()
                x0=tmp.get_positions().flatten()
                vv=VCG(tmp.get_chemical_symbols(),masses=mm)
                start=len(icSystem(vv(x0),len(tmp), masses=mm,xyz=x0).getStretchBendTorsOop()[0][0])
            if nummodes>numvec-start:
                nummodes=numvec-start
            w=np.random.choice(range(start,numvec),size=nummodes,replace=False)
            for i in w:
                disp[ads1:ads2,:3]+=vectors[i]*np.random.uniform(-1.,1.) #this is important if there is an adsorbate.
    
            disp/=np.max(np.abs(disp))
            rn=ro+self.stepwidth*disp
            
            if self.adjust_cm:
                cmt = atms.get_center_of_mass()
                
            atms.set_positions(rn)
    
            if self.adjust_cm is not None:
                cm = atms.get_center_of_mass()
                atms.translate(cmt - cm)
        elif self.ads:
            cc=Celltrans(self.stepwidth,self.adsorbate,self.cell_scale,self.adjust_cm)
            atms=cc.displace(atms)
        else:
            cc=Cartesian(self.stepwidth,self.adsorbate,self.adjust_cm)
            atms=cc.displace(atms)
        return atms
    
    def print_params(self):
        if self.ads:
            ads=', %d:%d is displaced'%(self.adsorbate[0],self.adsorbate[1])
        else:
            ads=''
        if self.constrain:
            cc=''
        else:
            cc=' not'
        return '%s: stepwidth=%f, numdelocmodes=%f%s, stretches are%s constrained'%(self.__class__.__name__,self.stepwidth,self.numdelmodes,ads,cc)
    
class Celltrans(Displacer):
    """displaces along cell vectors"""
    def __init__(self,stepwidth,adsorbate,cell_scale=[1.,1.,1.],adjust_cm=True):
        Displacer.__init__(self)
        self.stepwidth=stepwidth
        self.adsorbate=adsorbate
        self.cell_scale=cell_scale
        self.adjust_cm=adjust_cm
        
    def displace(self, tmp):
        ro=tmp.get_positions()
        disp=np.zeros((len(ro),3))
        c=tmp.get_cell()*self.cell_scale
        disp[self.adsorbate[0]:self.adsorbate[1],:3] = np.dot(c,np.random.uniform(-1., 1., 3))
        rn = ro + self.stepwidth * disp
        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        tmp.set_positions(rn)
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)  
        return tmp
    
    def print_params(self):
        ads=', %d:%d is displaced'%(self.adsorbate[0],self.adsorbate[1])
        return '%s: stepwidth=%f%s'%(self.__class__.__name__,self.stepwidth,ads)
    
class Cartesian(Displacer):
    def __init__(self,stepwidth,adsorbate=None,adjust_cm=True):
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
        else:
            self.adsorbate=adsorbate
            self.ads=True
        self.adjust_cm=adjust_cm
        self.stepwidth=stepwidth
    
    def displace(self, tmp):
        """Move atoms by a random step."""
        ro=tmp.get_positions()
        disp=np.zeros((len(ro),3))
        if self.ads:
            disp[self.adsorbate[0]:self.adsorbate[1],:3] = np.random.uniform(-1., 1., (self.adsorbate[1]-self.adsorbate[0], 3))
        else:
            disp[:,:3] = np.random.uniform(-1., 1., (len(tmp), 3))
        rn = ro + self.stepwidth * disp
        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        tmp.set_positions(rn) 
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)   
        return tmp
    
    def print_params(self):
        if self.ads:
            ads=', %d:%d is displaced'%(self.adsorbate[0],self.adsorbate[1])
        else:
            ads=''
        return '%s: stepwidth=%f%s'%(self.__class__.__name__,self.stepwidth,ads)

class Remove(Displacer):
    def __init__(self,prob=0.5,adsorbate=None,adjust_cm=True):
        '''Work in progress: here adsorbate must be an integer. Atoms tagged with it belong to the adsorbate'''
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
        else:
            self.adsorbate=adsorbate ## INT!!
            self.ads=True
        self.adjust_cm=adjust_cm
        self.prob=prob
    
    def displace(self, tmp):
        """Randomly remove atom with probability=prob"""
        '''CP sorting displaced atoms (or adsorbate only) is needed otherwise ase refuses to write trajectories if the composition is the same but the ordering of atoms differs, i.e. after insertion+removal+insertion. Sorting every time the composition changes should be enough'''
        if np.random.random() < self.prob:
            if self.ads:
                ads=Atoms([atom for atom in tmp if atom.tag==self.adsorbate])         ### separate
                slab=Atoms([atom for atom in tmp if not atom.tag==self.adsorbate],pbc=tmp.pbc,cell=tmp.cell,constraint=tmp.constraints)
                ads.pop(np.random.choice(ads).index)                                      ### pop atom from ads
                tmp=slab.extend(sort(ads))                                                ### put back together
            else:
                tmp.pop(np.random.choice(tmp).index)
                tmp=sort(tmp)
        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)   
        return tmp
    
    def print_params(self):
        if self.ads:
            ads=', %d: is displaced',(self.adsorbate)
        else:
            ads=''
        return '%s: probability=%f%s'%(self.__class__.__name__,self.prob,ads)

class Insert(Displacer):
    def __init__(self,prob=0.5,adsorbate=None,adjust_cm=True,sys_type='cluster'):
        '''Work in progress: here adsorbate must be an integer. Atoms tagged with it belong to the adsorbate'''
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
        else:
            self.adsorbate=adsorbate
            self.ads=True
        self.adjust_cm=adjust_cm
        self.prob=prob
        self.sys_type=sys_type

    def randomize(self,tmp):
        """Generates coordinates of new atom on the surface of the sphere enclosing the current structure. Works well with clusters.
        """
        com=tmp.get_center_of_mass()
        tmp.translate(-com)
        R=max(np.linalg.norm(tmp.get_positions(), axis=1))
        R=1.5*R ## should alleviate overlaps
        u = np.random.uniform(0,1)
        v = np.random.uniform(0,1)
        theta = 2*np.pi*u
        phi = np.arccos(2*v-1)
        x = R*np.cos(theta)*np.cos(phi)
        y = R*np.cos(theta)*np.sin(phi)
        z = R*np.sin(theta)
        return com+(x,y,z) #### find a smart way to do this for either cluster or adsorbate

    def random_duplicate(self,tmp):
        """pick a random atom and duplicate it, shifting it a bit farther from the COM --- not working atm"""
        tmp.translate(-tmp.get_center_of_mass())
        newatom=tmp[-1]#np.random.choice(tmp)
        newatom.position*=1.5
        return newatom.position
        
    def displace(self, tmp):
        """Randomly insert atom with probability=prob. Atom is chosen randomly from the composition."""
        atm=np.random.choice(Stoichiometry().get(tmp).keys())
        if np.random.random() < self.prob:
            if self.ads:
       #         if self.sys_type is 'cluster':
                    ads=Atoms([atom for atom in tmp if atom.tag==self.adsorbate])         ### separate
                    slab=Atoms([atom for atom in tmp if not atom.tag==self.adsorbate],pbc=tmp.pbc,cell=tmp.cell,constraint=tmp.constraints)
                    newpos=self.randomize(ads.copy())
                    ads.append(Atom(atm,newpos))
                    tmp=slab.extend(sort(ads))                                                ### put back together
            else:
        #        if self.sys_type is 'cluster':
                    newatom=self.randomize(tmp.copy())
                    tmp.append(Atom(atm,newatom))
                    tmp=sort(tmp)
        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)   
        return tmp
    
    def print_params(self):
        if self.ads:
            ads=', %d: is displaced',(self.adsorbate)
        else:
            ads=''
        return '%s: probability=%f%s'%(self.__class__.__name__,self.prob,ads)

class GC(Displacer):
    """Grand Canoical moves: insert or removeparticle with probability p, otherwise displace in DICs"""
    def __init__(self,prob=0.5,stepwidth=1.0,numdelocmodes=1,constrain=False,adsorbate=None,cell_scale=[1.0,1.0,1.0],adjust_cm=True,periodic=False):
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
        else:
            self.adsorbate=adsorbate
            self.ads=True
        self.prob=prob
        self.constrain=constrain
        self.stepwidth=stepwidth
        self.numdelmodes=np.abs(numdelocmodes)
        self.periodic=periodic
        self.adjust_cm=adjust_cm

    def displace(self, tmp):
        """Randomly insert atom with probability=prob"""
        if np.random.random() < self.prob:
            if np.random.random() < 0.5:  ##toss coin insert or remove
                print 'inserting'
                disp=Insert(prob=1,adsorbate=self.adsorbate)
            else:  ##toss coin insert or remove
                print 'removing'
                disp=Remove(prob=1,adsorbate=self.adsorbate)
            tmp=disp.displace(tmp)
        else:
            #print 'displacing by '+str(self.stepwidth)
            disp=DI(self.stepwidth,numdelocmodes=self.numdelmodes,constrain=self.constrain)
            tmp=disp.displace(tmp)
        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)   
        return tmp
    
    def print_params(self):
        if self.ads:
            ads=', %d: is displaced',(self.adsorbate)
        else:
            ads=''
        return '%s: probability=%f%s'%(self.__class__.__name__,self.prob,ads)

