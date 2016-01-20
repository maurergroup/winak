from abc import ABCMeta, abstractmethod
from ase.all import *
import numpy as np
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.InternalCoordinates import icSystem
from winak.globaloptimization.disspotter import DisSpotter
import os
from samba.policy import ads_to_dir_access_mask

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
        return '%s: stepwidth=%f%s'%(self.__class__.__name__,self.stepwidth,ads)
        
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
        """No DIS for atoms! Atoms get Cart movements"""
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
        else:
            cc=Cartesian(self.stepwidth,self.adsorbate,self.adjust_cm)
            atms=cc.displace(atms)
        return atms
    
    def print_params(self):
        if self.ads:
            ads=', %d:%d is displaced',(self.adsorbate[0],self.adsorbate[1])
        else:
            ads=''
        return '%s: stepwidth=%f, numdelocmodes=%f%s'%(self.__class__.__name__,self.stepwidth,self.numdelmodes,ads)
    
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
            ads=', %d:%d is displaced',(self.adsorbate[0],self.adsorbate[1])
        else:
            ads=''
        return '%s: stepwidth=%f%s'%(self.__class__.__name__,self.stepwidth,ads)