from abc import ABCMeta, abstractmethod
from ase.all import *
from winak.screening.composition import Stoichiometry
from ase.utils.geometry import sort

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
        EE.__init__(self, calc,optlog)
        self.fmax=fmax
        self.opt=opt
        self.fmax2=fmax2
        self.opt2=opt2
        
    def get_energy(self, atoms):
        """If it hasn't converged after 3000 steps, it probably won't ever"""
        ret=None
        try:
            atoms.set_calculator(self.calc)
            opt = self.opt(atoms,logfile=self.optlog)
            opt.run(fmax=self.fmax,steps=3000)
            if opt.converged() and self.opt2 is not None and self.fmax2 is not None:
                opt=self.opt2(atoms,logfile=self.optlog)
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

class grandEE(EE):
    """grandcanonical potential energy optimization. evaluates the free energy of formation for gas-phase molecules or the surface free energy per unit area 
    for surface adsorptions/reconstructions
    starting with opt until fmax and then followed by opt2 until fmax2 is reached. Use of 2 optimizers is optional"""
    def __init__(self,calc,opt,fmax,opt2=None,fmax2=None,optlog='opt.log',ecomp={},mu={}):
        EE.__init__(self, calc,optlog)
        self.fmax=fmax
        self.opt=opt
        self.fmax2=fmax2
        self.opt2=opt2        
        self.ecomp=ecomp
        self.mu=mu
        
    def get_energy(self, atoms):
        """comp is the composition array (i.e. n_i), mu is the array of chemical potentials,
        ecomp is the array of the total energy of components (e.g. atomici) DICTIONARIES"""
        """If it hasn't converged after 3000 steps, it probably won't ever"""
        atoms=sort(atoms) ##must be sorted for traj
        composition=Stoichiometry()
        stoich=composition.get(atoms)
        #####CP make arrays with composition, free atom energies and chemical potential, sorted by atomic symbols, from dictionaries
        comp=[stoich[c] for c in sorted(stoich)]
        ecomp=[self.ecomp[e] for e in sorted(self.ecomp)]
        mu=[self.mu[m] for m in sorted(self.mu)]
        ret=None
        try:
            atoms.set_calculator(self.calc)
            opt = self.opt(atoms,logfile=self.optlog)
            opt.run(fmax=self.fmax,steps=3000)
            if opt.converged() and self.opt2 is not None and self.fmax2 is not None:
                opt=self.opt2(atoms,logfile=self.optlog)
                opt.run(fmax=self.fmax2,steps=3000)
            if opt.converged():
                #print 'fin qui tutto bene'
                EE=atoms.get_potential_energy()
                #print EE, comp
                grandE=EE-np.dot(comp,ecomp)-np.dot(comp,mu)
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
