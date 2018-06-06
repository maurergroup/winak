# winak.screening.criterion
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
#from winak.constants import kB
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

class GCMetropolis(Criterion):
    """Just copied and pasted for now, actually should work out of the box if the energy evaluator is correct."""
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


class PopulationSelection:
    """This class selects the structures of a population that are accepted in the following generation."""
    __metaclass__ = ABCMeta

    def __init__(self):
        """subclasses must call this method."""
        pass
        
    @abstractmethod
    def filter(self,pop,popsize):
        """subclasses must implement this method. Has to return a filtered population"""
        pass
    
    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing
        all the important parameters"""
        pass

class FabioSelection(PopulationSelection):
    """Accepts the n=popsize best structures in the population, according to the fitness parameter found in info"""
    def __init__(self,popsize):
        PopulationSelection.__init__(self)
        self.popsize = popsize
    def filter(self,pop):
        newmut = 0
        newmat = 0
        SortedPopulation = sorted(pop, key=lambda x: x.info["fitness"], reverse=True)  ###higher fitness comes FIRST
        FilteredPopulation,diversity,energy = self._select_best_subset(SortedPopulation)
        diversity = 0

        for structure in FilteredPopulation:
            if hasattr(structure,"info"):
                if "New" in structure.info:
                    if structure.info["New"]:
                        if "origin" in structure.info:
                            if structure.info["Origin"] == "Mutated":
                                newmut += 1
                            elif structure.info["Origin"] == "Child":
                                newmat += 1
        report = "\n"+"New structures accepted in the next generation:"+str(newmat+newmut)+"\n"+"Generated by mating:"+str(newmat)+"\n"+"Generated by mutation:"+str(newmut)+"\n"
        counter = 1
        for structure in FilteredPopulation:
            description ="\n"+ "%s: Fitness = %s" %(str(counter),structure.info["fitness"])
            if hasattr(structure,"info"):
                if "New" in structure.info:
                    if structure.info["New"]:
                        description += " <<< NEW - %s, from previous generation [%s]"%(structure.info["Origin"],structure.info["Ascendance"])
                    report += description
                    counter += 1
        return FilteredPopulation,report,diversity
    
    def _select_best_subset(self,pop):
        subset = pop[:self.popsize]
        diversity = 0
        energy = None
        return subset,diversity,energy


    def print_params(self):
        return "Accepts the best structures according to fitness"
