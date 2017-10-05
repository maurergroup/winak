# winak.screening.energyevaluator
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
from winak.screening.composition import Stoichiometry

### CP - ugly hack to deal with ASE backwards compatibility; this is temporary
try:
    from ase.build.tools import sort
except:
    from ase.geometry import sort
###

from ase.atoms import Atoms

class EE:
    """This class relaxes an ase.atoms object in any way you see fit. The only
    restriction is, that it must have a get_energy and a print_params method as
    described below."""
    __metaclass__ = ABCMeta

    def __init__(self,calc,optlog='opt.log'):
        """subclasses must call this method. If the number of atoms changes, the
        calculator has to be reset."""
        self.optlog=optlog
        self.calc=calc
        
    @abstractmethod
    def get_energy(self,atoms):
        """subclasses must implement this method. Has to return a list containting 
        the optimized ase.atoms object and then the energy or None if something went wrong"""
        pass
    
    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing
        all the important parameters"""
        pass
    
class potEE(EE):
    """standard potential energy optimization, starting with opt until fmax
    and then followed by opt2 until fmax2 is reached. Use of 2 optimizers is optional"""
    def __init__(self,calc,opt,fmax,opt2=None,fmax2=None,optlog='opt.log'):
        EE.__init__(self, calc, optlog)
        self.fmax=fmax
        self.opt=opt
        self.fmax2=fmax2
        self.opt2=opt2
        
    def get_energy(self, atoms):
        """If it hasn't converged after 3000 steps, it probably won't ever"""
        ret=None
        try:
            atoms.set_calculator(self.calc)
            opt = self.opt(atoms,logfile=self.optlog,trajectory='relax.traj')
            opt.run(fmax=self.fmax,steps=3000)
            if opt.converged() and self.opt2 is not None and self.fmax2 is not None:
                opt=self.opt2(atoms,logfile=self.optlog,trajectory='relax.traj')
                opt.run(fmax=self.fmax2,steps=3000)
            if opt.converged():
                ret=(atoms,atoms.get_potential_energy())
        except:
            """Something went wrong."""                
            ret=None
        return ret
    
    def print_params(self):
        ret='%s: Optimizer=%s, fmax=%f'%(self.__class__.__name__,self.opt.__name__,self.fmax)
        if self.opt2 is not None and self.fmax2 is not None:
            ret+=', Optimizer2=%s, fmax2=%f '%(self.opt2.__name__,self.fmax2)
        return ret

class grandEE(potEE):
    """grandcanonical potential energy optimization. evaluates the free energy of formation for gas-phase molecules or the surface free energy per unit area 
    for surface adsorptions/reconstructions
    starting with opt until fmax and then followed by opt2 until fmax2 is reached. Use of 2 optimizers is optional"""
    def __init__(self,calc,opt,fmax,opt2=None,fmax2=None,optlog='opt.log',ecomp={},mu={},eref=0.0,adsorbate=None):
        potEE.__init__(self, calc, opt, fmax, opt2, fmax2, optlog)
        #self.fmax=fmax
        #self.opt=opt
        #self.fmax2=fmax2
        #self.opt2=opt2        
        self.ecomp=ecomp
        self.mu=mu
        self.eref=eref  ## clean surface energy for adsorbates
        self.adsorbate=adsorbate
        
    def get_energy(self, atoms):
        """comp is the composition array (i.e. n_i), mu is the array of chemical potentials,
        ecomp is the array of the total energy of components (e.g. atomic) DICTIONARIES"""
        """If it hasn't converged after 3000 steps, it probably won't ever"""
        #atoms=sort(atoms) ##must be sorted for traj --- no, sorting in displacements is enough
        composition=Stoichiometry()
        if self.adsorbate is not None:
            ads=Atoms([atom for atom in atoms if atom.tag==self.adsorbate])
            stoich=composition.get(ads)
        else:
            stoich=composition.get(atoms)
        #####CP make arrays with composition, free atom energies and chemical potential, sorted by atomic symbols, from dictionaries
        comp=[stoich[c] for c in sorted(stoich)]
        #ecomp=[self.ecomp[e] for e in sorted(self.ecomp)]
        ecomp=[self.ecomp[e] for e in sorted(stoich)]
        #mu=[self.mu[m] for m in sorted(self.mu)]
        mu=[self.mu[m] for m in sorted(stoich)]
        ret=None
        try:
            atoms.set_calculator(self.calc)
            ## CP the following fixes some back-compatibility issue with ASE; since v.3.13 the force_consistent tag was introduced in optimizers BUT not all of them
            try:
                opt = self.opt(atoms,logfile=self.optlog,force_consistent=False, trajectory='relax.traj')
            except:
                opt = self.opt(atoms,logfile=self.optlog, trajectory='relax.traj')
            opt.run(fmax=self.fmax,steps=3000)
            if opt.converged() and self.opt2 is not None and self.fmax2 is not None:
                opt=self.opt2(atoms,logfile=self.optlog, trajectory='relax.traj')
                opt.run(fmax=self.fmax2,steps=3000)
            if opt.converged():
                import numpy as np
                #print 'fin qui tutto bene'
                EE=atoms.get_potential_energy()
                grandE=EE-np.dot(comp,ecomp)-np.dot(comp,mu)-self.eref
                ret=(atoms,grandE)
        except:
            """Something went wrong."""                
            #print 'Something went wrong.'              
            ret=None
        return ret
    
    def print_params(self):
        ret='%s: Optimizer=%s, fmax=%f'%(self.__class__.__name__,self.opt.__name__,self.fmax)
        if self.opt2 is not None and self.fmax2 is not None:
            ret+=', Optimizer2=%s, fmax2=%f '%(self.opt2.__name__,self.fmax2)
	return ret
