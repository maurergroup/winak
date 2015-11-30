from abc import ABCMeta, abstractmethod
from ase.all import *
from ase.units import kB
import numpy as np

class Criterion:
    """This class accepts or declines a step in any way you see fit."""
    __metaclass__ = ABCMeta

    def __init__(self):
        """subclasses must call this method."""
        pass
        
    @abstractmethod
    def evaluate(self,tmp):
        """subclasses must implement this method. Has to return a boolean if
        tmp is accepted"""
        pass
    
    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing
        all the important parameters"""
        pass
    
class Metropolis(Criterion):
    def __init__(self,T=100*kB,energy='pot'):
        """if you don't want the potential energy, just change the first if in
        evaluate by adding your own elif"""
        CR.__init__(self)
        self.kT=T
        self.emin=None
        self.amin=None
        self.minima=[]
        self.energies=[]
        self.energy=energy
        
    def evaluate(self, tmp):
        ret=False
        if self.energy=='pot':
            En=tmp.get_potential_energy()
        self.energies.append(En)
        self.minima.append(tmp.copy())
        if self.emin is None or En<self.emin:
            self.emin=En
            ret=True
        elif np.exp((Eo - En) / self.kT) > np.random.uniform():
            ret=True
        return ret
    
    def print_params(self):
        return '%s: T=%f, energy='%(self.__class__.__name__,self.kT/kB,self.energy)