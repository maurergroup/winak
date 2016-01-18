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
    def evaluate(self,tmp,en):
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
        Criterion.__init__(self)
        self.kT=T
        self.emin=None
        self.amin=None
        self.Eo=None #lastaccepted E
        self.minima=[]
        self.energies=[]
        self.energy=energy
        
    def evaluate(self, tmp, en):
        ret=False
        self.energies.append(en)
        self.minima.append(tmp.copy())
        if self.emin is None or en<self.emin:
            self.emin=en
            self.Eo=en
            ret=True
        elif np.exp((self.Eo - en) / self.kT) > np.random.uniform():
            ret=True
            self.Eo=en
        return ret
    
    def print_params(self):
        return '%s: T=%f, energy=%s'%(self.__class__.__name__,self.kT/kB,self.energy)