# winak.screening.displacer
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

from abc import ABCMeta, abstractmethod

### CP - ugly hack to deal with ASE backwards compatibility; this is temporary
try:
    from ase.build.tools import sort
except:
    from ase.utils.geometry import sort
try:
    from ase.neighborlist import NeighborList
except:
    from ase.calculators.neighborlist import NeighborList
###

from ase.atom import Atom
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
    """This class performs a step in any way you see fit. It is also 
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
    def __init__(self,stepwidth,numdelocmodes=1,constrain=False,adsorbate=None,cell_scale=[1.0,1.0,1.0],adjust_cm=True,periodic=False,dense=True,loghax=False):
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
        self.dense=dense
        self.adjust_cm=adjust_cm
        self.stepwidth=stepwidth
        self.numdelmodes=np.abs(numdelocmodes)
        self.lh=loghax
        self.periodic=periodic
        
    def displace(self,tmp):
        atms=tmp.copy()
        if self.ads:
            ads=tmp[self.adsorbate[0]:self.adsorbate[1]].copy()
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
            
            di=DI(self.stepwidth,numdelocmodes=self.numdelmodes,constrain=self.constrain,adsorbate=adstmp,cell_scale=self.cell_scale,adjust_cm=self.adjust_cm,periodic=self.periodic,dense=self.dense)
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
            ads=', %d:%d is displaced'%(self.adsorbate[0],self.adsorbate[1])
        else:
            ads=''
        if self.constrain:
            cc=''
        else:
            cc=' not'
        return '%s: stepwidth=%f%s, stretches are%s constrained'%(self.__class__.__name__,self.stepwidth,ads,cc)
        
class DI(Displacer):
    def __init__(self,stepwidth,numdelocmodes=1,constrain=False,adsorbate=None,cell_scale=[1.0,1.0,1.0],adjust_cm=True,periodic=False,dense=True,weighted=True,thresholds=[2.5,0,0]):
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
        self.dense=dense
        self.weighted=weighted
        self.periodic=periodic
        self.thresholds=thresholds
        
    def get_vectors(self,atoms):
        if self.periodic:
            deloc=Delocalizer(atoms,periodic=True,
                    dense=self.dense,
                    weighted=self.weighted,
                    threshold=self.thresholds[0],
                    bendThreshold=self.thresholds[1],
                    torsionThreshold=self.thresholds[2])
            coords=PC(deloc.x_ref.flatten(), deloc.masses, atoms=deloc.atoms, ic=deloc.ic, Li=deloc.u)
        else:
            deloc=Delocalizer(atoms,dense=self.dense,
                    weighted=self.weighted,
                    threshold=self.thresholds[0],
                    bendThreshold=self.thresholds[1],
                    torsionThreshold=self.thresholds[2])
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
            #w=[0] #np.random.choice(range(start,numvec),size=nummodes,replace=False)
            #print 'w = ', w, 'out of ', numvec
            for i in w:
                disp[ads1:ads2,:3]+=vectors[i]*np.random.uniform(-1.,1.) #this is important if there is an adsorbate.
            #print disp
            disp/=np.max(np.abs(disp))
            rn=ro+self.stepwidth*disp
            
            if self.adjust_cm:
                cmt = atms.get_center_of_mass()
                
            atms.set_positions(rn)
            
            if self.adjust_cm:         
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
    def __init__(self,prob=0.5,adsorbate=None,adjust_cm=True,atm=None):
        '''Work in progress: here adsorbate must be an integer. Atoms tagged with it belong to the adsorbate'''
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
        else:
            self.adsorbate=adsorbate ## INT!!
            self.ads=True
        self.adjust_cm=adjust_cm
        self.prob=prob
        self.atm=atm
            
    def pop_atom(self, atoms, slab=None):
        if self.ads:
            ads=Atoms([atom for atom in atoms if atom.tag==self.adsorbate])         ### separate
            tmp=ads.copy()
            slab=Atoms([atom for atom in atoms if not atom.tag==self.adsorbate],pbc=atoms.pbc,cell=atoms.cell,constraint=atoms.constraints)
        else:
            tmp=atoms.copy()
        if self.atm is None:
            idx=np.random.choice(tmp).index                                     ### pop atom from ads
        else:
            idx=np.random.choice([atom.index for atom in tmp if atom.symbol==self.atm])
        nspecies=len(Stoichiometry().get(tmp).keys())          ### to avoid disappearing species
        tmp.pop(idx)                                                              ### pop atom from ads
        tmp=sort(tmp)    
        return tmp, nspecies, slab          ## will put back together in displace subclass
    
    def displace(self, tmp, ok=False):
        """Randomly remove atom with probability=prob"""
        #CP sorting displaced atoms (or adsorbate only) is needed otherwise ase refuses to write trajectories if the composition is the same but the ordering of atoms differs, i.e. after insertion+removal+insertion. Sorting every time the composition changes should be enough
        if np.random.random() < self.prob:
            tries=0
            while not ok:
                pippo,nspecies,slab=self.pop_atom(tmp.copy())
                try:
                    ok=(len(Stoichiometry().get(pippo).keys())==nspecies) ## check if atomic species disappears; also includes the case of running out of atoms completely 
                except:                                                     ### though maybe that should be treated separately and break, to avoid doing this 10 times
                    print 'atomic species disappeared'
                tries+=1
                #print tries
                if tries>3:
                    return None
                    break
            if self.ads:
                tmp=slab.extend(pippo)                                                ### put back together
                #tmp.constraints[0].delete_atom(-1)  ## remove one boolean value from the constraints to avoid traj complaining; not sure -1 is ok OR the true index shall be passed
                ### seems so be no longer needed in latest ASE
            else:
                tmp=pippo
        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)   
        return tmp
    
    def print_params(self):
        if self.ads:
            ads=', %d: is displaced'%(self.adsorbate)
        else:
            ads=''
        return '%s: probability=%f%s'%(self.__class__.__name__,self.prob,ads)

class Insert(Displacer):
    def __init__(self,prob=0.5,adsorbate=None,adjust_cm=True,sys_type='cluster',mode='nn',atm=None):
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
        self.mode=mode
        self.atm=atm

    def nn(self,tmp,cutoff=1.5):
        """Picks a random atom and its nearest neighbor and puts new atom somewhere in between"""
        com=tmp.get_center_of_mass()
        tmp.translate(-com)
        cutoffs=np.full(len(tmp),cutoff) ## TODO:read from covalent radii data 
        nl=NeighborList(cutoffs, bothways=True)
        nl.build(tmp)
        pippo=np.random.choice(tmp).index
        pluto=nl.get_neighbors(pippo)[0][1] ## index of nearest neighbour. Mind: [0][0] is the atom itself
        #new=(tmp.positions[pippo]+tmp.positions[pluto])/2+[0.,0.,cutoff] ## push a bit up along z, for adsorbates TODO push away from COM rather than up
        new=(tmp.positions[pippo]+tmp.positions[pluto])/2
        new=new*(1+cutoff/np.linalg.norm(new))
        return com+new

    def randomize(self,tmp):
        """Generates coordinates of new atom on the surface of the sphere enclosing the current structure. Works well with clusters.
        """
        com=tmp.get_center_of_mass()
        tmp.translate(-com)
        R=max(np.linalg.norm(tmp.get_positions(), axis=1))
        R = np.random.uniform(0,R) ## should be crap, use for performance evaluation (as a bad example)
        u = np.random.uniform(0,1)
        v = np.random.uniform(0,1)
        theta = 2*np.pi*u
        phi = np.arccos(2*v-1)
        x = R*np.cos(theta)*np.cos(phi)
        y = R*np.cos(theta)*np.sin(phi)
        z = R*np.sin(theta)
        return com+(x,y,z) #### find a smart way to do this for either cluster or adsorbate
    
    def randomize_surf(self,tmp):
        """Generates coordinates of new atom on the surface of the sphere enclosing the current structure. Works well with clusters.
        """
        com=tmp.get_center_of_mass()
        tmp.translate(-com)
        R=max(np.linalg.norm(tmp.get_positions(), axis=1))
        #R = (1+1/R)*R ## should alleviate overlaps
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
        """Randomly insert atom with probability=prob. Atom type is chosen randomly from the composition."""
        if np.random.random() < self.prob:
            if self.ads:
                ads=Atoms([atom for atom in tmp if atom.tag==self.adsorbate])         ### separate
                slab=Atoms([atom for atom in tmp if not atom.tag==self.adsorbate],pbc=tmp.pbc,cell=tmp.cell,constraint=tmp.constraints)
                tmp=ads.copy()
            
            if self.atm is None:
                atm=np.random.choice(Stoichiometry().get(tmp).keys())
            else:
                atm = self.atm
            
            if self.mode=='nn': 
                newpos=self.nn(tmp.copy())
            elif self.mode=='rsphere':
                newpos=self.randomize(tmp.copy())
            elif self.mode=='rsurf':
                newpos=self.randomize_surf(tmp.copy())
            else:
                raise ValueError('Invalid insertion mode')
            if self.ads:
                tmp.append(Atom(atm,newpos,tag=100))
                tmp=slab.extend(sort(tmp))                                                ### put back together
            else:
                tmp.append(Atom(atm,newpos))
                tmp=sort(tmp)

        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)   
        return tmp
    
    def print_params(self):
        if self.ads:
            ads=', %d: is displaced'%(self.adsorbate)
        else:
            ads=''
        return '%s: probability=%f%s'%(self.__class__.__name__,self.prob,ads)

class GC(Displacer):
    """Grand Canonical moves: insert or remove particle with probability prob, otherwise displace in DICs"""
    def __init__(self,prob=0.5,stepwidth=1.0,numdelocmodes=1,constrain=False,adsorbate=None,cell_scale=[1.0,1.0,1.0],adjust_cm=True,periodic=False,bias=0.5,ins_mode='nn',atm=None):
        Displacer.__init__(self)
        if adsorbate is None:
            self.ads=False
            self.adsorbate=adsorbate
        else:
            self.adsorbate=adsorbate
            self.ads=True
        self.prob=prob
        self.constrain=constrain
        self.stepwidth=stepwidth
        self.numdelmodes=np.abs(numdelocmodes)
        self.periodic=periodic
        self.adjust_cm=adjust_cm
        self.bias=bias ## bias towards either insertion or removal (I/R ratio)
        self.ins_mode=ins_mode
        self.atm=atm
    def displace(self, tmp):
        """Randomly insert atom with probability=prob"""
        if np.random.random() < self.prob:
            if np.random.random() < self.bias:  ##toss coin insert or remove or bias towards either
                disp=Insert(prob=1,adsorbate=self.adsorbate,adjust_cm=self.adjust_cm,mode=self.ins_mode,atm=self.atm)
                #print 'inserting'
            else: 
                disp=Remove(prob=1,adsorbate=self.adsorbate,adjust_cm=self.adjust_cm,atm=self.atm)
                #print 'removing'
            tmp=disp.displace(tmp)
        else:
            if self.ads:    ## cannot be done in init bc it has to be updated... find a way
                adsidx=[atom.index for atom in tmp if atom.tag==self.adsorbate]    ### convert adsorbate by tag to adsorbate by index
                #print 'ads = ', (adsidx[0],adsidx[-1]+1)
                disp=DI(self.stepwidth,numdelocmodes=self.numdelmodes,constrain=self.constrain, adjust_cm=self.adjust_cm, adsorbate=(adsidx[0],adsidx[-1]+1), periodic=self.periodic, cell_scale=self.cell_scale)
                #print 'displacing by '+str(self.stepwidth)+', ads = '+str(adsidx[0])+','+str(adsidx[-1]+1)
            else:
                disp=DI(self.stepwidth,numdelocmodes=self.numdelmodes,constrain=self.constrain, adjust_cm=self.adjust_cm, adsorbate=None)#self.adsorbate)
                #print 'displacing by '+str(self.stepwidth)
            tmp=disp.displace(tmp)
        if self.adjust_cm:
            cmt = tmp.get_center_of_mass()  
        if self.adjust_cm:
            cm = tmp.get_center_of_mass()
            tmp.translate(cmt - cm)   
        return tmp
    
    def print_params(self):
        if self.ads:
            ads=', %d: is displaced'%(self.adsorbate)
        else:
            ads=''
        return '%s: probability=%f%s'%(self.__class__.__name__,self.prob,ads)


class PopulationGenerator:
    """This class generates a population of structures, starting from a generic input"""
    __metaclass__ = ABCMeta

    def __init__(self):
        """subclasses must call this method."""
        pass

    @abstractmethod
    def generate(self,input,popsize):
        """subclasses must implement this method. Has to return a population of structures."""
        pass

    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing all the imporant parameters."""
        pass

class OurPopulationGenerator(PopulationGenerator):
    def __init__(self):
        #still all to figure out

    def generate(self,input,popsize):
        #still all to figure out

    def print_params(self):
        #still all to figure out



class PopulationManager:
    """This class performs a step on a population of structures. Just like Displacer, it is also in charge of logging."""
    __metaclass__ = ABCMeta

    def __init__(self):
        """subclasses must call this method."""
        pass

    @abstractmethod
    def evolve(self,pop):
        """subclasse must implement this method. Has to return a modified population of structures."""
        pass

    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing all the important parameters."""
        pass


class FabioManager(PopulationManager):
    def __init__(self,MatingManager,MatingParameters,MutationManager,MutationParameters,Xparameter): #to add: parameters for following classes  
    """Performs an evolution step on a given population. Distributes work to the Mating and (if desired) Mutation classes, according to the Xparameter."""   
       PopulationManager.__init__(self)
       self.MatingManager=MatingManager
       self.MatingParameters=MatingParameters
       self.MutationManager=MutationManager
       self.MutationParameters=MutationParameters
       self.Xparameter=np.abs(Xparameter)


   def evolve(self,pop):
       #distribute work to MatingManager and MutationManager
       #should receive new two pop objects
       Xparameter = self.Xparameter

       MatingManager = self.MatingManager(MatingParameters)
       OffspringStructures = self.MatingManager.MatePopulation(pop,Xparameter)
       
       MutationManager = self.MutationManager(MutationParameters)
       MutatedStructures = MutationManager.MutatePopulation(pop,Xparameter)
        
       #generates the evolved population merging parents, offspring and mutated 
       newpop=[]
       for stru in pop:
           newpop.append(stru.copy())

       for stru in OffspringStructures:
           newpop.append(stru.copy())

       for stru in MutatedStructures:
           newpop.append(stru.copy())
       #to be implemented: option to avoid overwriting
       write('newpop.traj',newpop)
       newpop = Trajectory('newpop.traj','r')
       return newpop

    def print_params(self):
        #to implement
        return "" 



class MatingManager:
    """This class performs matings on a population.
    Receives the population, performs matings according to Xparameter and the selected MatingOperator, and returns the offspring population"""
    __metaclass__ = ABCMeta

    def __init__(self):
        """subclasses must call this method"""
        pass

    @abstractmethod
    def MatePopulation(self,pop):
        """subclasses must implement this method. Return an offspring of structures"""
        pass

    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing all the important parameters."""
        pass


class FabioMating(MatingManager):
    def __init__(self,MatingParameters): #to add: parameters for following classes
        MatingManager.__init__(self)
        self.MatingOperator = MatingParameters["MatingOperator"]
        self.MatingOperatorParameters = MatingParameters[MatingOperatorParameters]

    def MatePopulation(self,pop,Xparameter):
        MatingOperatorParameters = self.MatingOperatorParameters
        MatingOperator = self.MatingOperator(**MatingOperatorParameters)
        offspring = []

        #creates a list of structures suitable for mating
        poptomate = []
        for stru in pop:
            poptomate.append(stru.copy())
        poptomate =  poptomate[0:Xparameter]
        
        #mates every structure with a second structure,randomly selected from the whole population
        for stru in poptomate:
            #selects random partner
            candidates = list(poptomate)
            candidates.remove(stru)            
            partnernumber = np.random.randint(0,len(candidates))
            partner = candidates[partnernumber]
            
            #performs mating
            Children = MatingOperator.Mate(stru,partner) #to implement
            for struc in Children:
                offspring.append(struc.copy())

        return offspring

    def print_params(self): 
        #to implement
        return ""



class MatingOperator:
    """This class performs a single mating. Receives two structures, and returns two 'children' structures"""
    __metaclass__ = ABCMeta

    def __init__(self):
        """subclasses must call this method"""
        pass

    @abstractmethod
    def Mate(self,partner1,partner2):
        """subclasses must implement this method. Returns two 'children' structures"""
        pass

    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing all the important parameters."""
        pass


class SinusoidalCut(MatingOperator):
    def __init__(self,parameters):
        #to implement
         #begin with something random and ugly
    def Mate(self,partner1,partner2):
        #to be implemented

    def print_params(self):
        #to be implemented
        return ""

class TestMating(MatingOperator):
    def __init__(self,MatingOperatorParameters):
        MatingOperator.__init__self()
        pass    #no actual parameter for this test mating operator

    def Mate(self,partner1,partner2):
        Children = [] 
        choice = np.random.rand() 
  
        child1 = Atoms() 
        child2 = Atoms() 
        if choice > 0.5: 
            for atom in partner1: 
                if atom.x > 7.5: 
                    child1.append(atom) 
                else: 
                    child2.append(atom) 
            for atom in partner2: 
                if atom.x < 7.5: 
                    child1.append(atom) 
                else: 
                    child2.append(atom) 
        else: 
            for atom in stru1: 
                if atom.y > 7.5: 
                    child1.append(atom) 
                else: 
                    child2.append(atom)
            for atom in stru2: 
                if atom.y < 7.5: 
                    child1.append(atom) 
                else: 
                    child2.append(atom) 
  
        Children.append(child1) 
        Children.append(child2) 
   
        return Children

    def print_params(self):
        return "TestMating print_params"

class MutationManager:
    """This class performs mutations on a population.
Receives a population, performs mutations according to the xParameter, and returns the population of mutated structures"""
    __metaclass__= ABCMeta

    def __init__(self):
        """subclasses must call this method"""
        pass

    @abstractmethod
    def MutatePopulation(self,pop):
        """subclasses must implement this method. Returns a population of mutated structures"""
        pass

    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing all the important parameters."""
        pass


class FabioMutation(MutationManager):
    def __init__(self,MutationParameters):
        ###########
        MutationManager.__init__(self)
        self.MutationOperator = MutationParameters["MutationOperator"]
        self.MutationOperatorParameters = MutationParameters["MutationOperatorParameters"]
    

    def MutatePopulation(self,pop,Xparameter): 
        MutationOperatorParams = self.MutationOperatorParameters
        Operator = self.MutationOperator(**MutationOperatorParams)
        
        MutatedStructures= []
        for structure in pop[xParameter:]:
            mutated = Operator.Mutate(structure)
            MutatedStructures.append(mutated)
        return MutatedStructures

    def print_params(self):
        ########
        return ""

class TestMutationOperator:
    def __init__(self,parameters):
        pass
    
    def Mutate(self,structure):
        materials = ['Ni','Co','Zn','Cu']
        num = np.random.randint(0,len(structure))
        newel = np.random.choice(materials)
        structure[num].symbol = newel
        return structure


