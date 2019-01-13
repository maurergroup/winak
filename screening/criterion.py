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
import itertools
import operator
from datetime import datetime
from winak.SOAP_interface import quantify_dissimilarity
from winak.SOAP_interface import sim_matrix
import pandas as pd


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

class FlexibleSelection(PopulationSelection):
    def __init__(self,popsize,fitness_preponderance=1):
    """ This class returns a subset of the provided population, of size=popsize. The subset is selected according to the criterion provided in the execution of the filter function: best_fitness, best_diversity or best_compromise_elite """
        PopulationSelection.__init__(self)
        self.popsize = popsize
        self.oldE = 0
        self.oldd = 0
        self.fitness_preponderance = fitness_preponderance

    def filter(self,pop,mode="best_compromise_elite"):
        """ Sorts the structures according to their fitness, calls the best_subset_selection function with a given 'mode' parameter, records the value of average fitness and diversity for the resulting subset, and returns the subset, the values of diversity, average fitness, and fitness of the best structures, together with a report and a numerical report """
        time0 = datetime.now()
        mode=mode
        newmut = 0
        newmat = 0
        # sorting
        SortedPopulation = sorted(pop, key=lambda x: x.info["fitness"], reverse=True)  ###higher fitness comes FIRST
        # subset selection
        FilteredPopulation,diversity,fitness,first_fitness = self._select_best_subset(SortedPopulation,mode)
        #reporting
        for structure in FilteredPopulation:
            if hasattr(structure,"info"):
                if "New" in structure.info:
                    if structure.info["New"]:
                        if "Origin" in structure.info:
                            if structure.info["Origin"] == "Mutated":
                                newmut += 1
                            elif structure.info["Origin"] == "Child":
                                newmat += 1
        report = "\n"+"New structures accepted in the next generation:"+str(newmat+newmut)+"\n"+"Generated by mating: "+str(newmat)+"\n"+"Generated by mutation: "+str(newmut)+"\n"
        counter = 1
        for structure in FilteredPopulation:
            description ="\n"+ "%s: Fitness = %s" %(str(counter),structure.info["fitness"])
            if hasattr(structure,"info"):
                if "New" in structure.info:
                    if structure.info["New"]:
                        description += " <<< NEW - %s, from previous generation [%s]"%(structure.info["Origin"],structure.info["Ascendance"])
                    report += description
                    counter += 1
        time2=datetime.now()
        report+="\n"+"\n"+"Selection time: "+str(time2-time0)

        self.oldE = fitness
        self.oldd = diversity

        num_report = pd.DataFrame({'Selection Time':(time2-time0),'Accepted new structures':(newmat+newmut),'Average F':fitness,'Best F':first_fitness,'Diversity':diversity},index=[3])
        return FilteredPopulation,report,diversity,fitness,first_fitness,num_report

    def _select_best_subset(self,pop,mode):
        ### Selects the best subset according to mode.
        ### mode = best_fitness   selects the self.popsize structures with highest fitness
        ###      = best_diversity   selects the subset that maximizes diversity among structures
        ###      = best_compromise_elite   always includes the structure with the highest fitness. Selects the subset that best improves both diversity and fitness 
        
        if len(pop)<=1 or self.popsize<=1:
            return (pop,0,pop[0].info["fitness"])
             
        if mode=="best_fitness":
            subset = pop[:self.popsize]
            diversity = quantify_dissimilarity(subset)
            energy = np.average(np.array([x.info["fitness"] for x in subset]))
        
        elif mode=="best_diversity":
            subset,diversity,energy = self._find_maximum_distance_subset(pop,self.popsize)

        elif mode=="best_compromise_elite":
            subset,diversity,energy = self._find_best_compromise_elite_subset(pop,self.popsize,self.fitness_preponderance)     
  
        first_fitness = subset[0].info['fitness']
        return subset,diversity,energy,first_fitness
    
    def _find_maximum_distance_subset(self,pop,popsize):
        """ Selects the subset of size popsize with the maximum average difference among structures, measured through SOAP indexes """
        # generates similarity matrix for the whole population
        similarity_matrix = sim_matrix(pop)
        # if the population size is smaller than popsize, directly returns the whole population
        if len(pop)<=popsize:
            subset=pop
            dissimilarity = quantify_dissimilarity(subset)
            energy = np.average(np.array([x.info["fitness"] for x in subset])) 
        else:
            # iterates through all possible subsets
            indexes = [n for n in range (len(pop))] 
            combins = itertools.combinations(indexes,popsize)
            subsets_distances = dict()
            for subset in combins:
                subset_list = list(subset)
                submatrix = similarity_matrix[np.ix_(subset_list,subset_list)]
                length = len(submatrix)
                higher_tri = submatrix[np.triu_indices(length,k=1)]
                vector_length = len(higher_tri)
                summation = sum(higher_tri)
                dissimilarity = (vector_length - summation)/vector_length
                subsets_distances[subset] = dissimilarity
            subset_numbers = max(subsets_distances.iteritems(),key=operator.itemgetter(1))[0]
            dissimilarity = subsets_distances[subset_numbers]
            subset = [pop[i] for i in subset_numbers]
            energy = np.average(np.array([x.info["fitness"] for x in subset])) 
        return subset,dissimilarity,energy

    def _find_best_compromise_elite_subset(self,pop,popsize,fitness_preponderance):
        """ Selects the subset of size popsize with the maximum value of Q, which expresses a balancement between average fitness and diversity. Only subsets including the best structure in the population are taken into consideration """
        # if the population size is smaller than popsize, directly returns the whole population
        if len(pop)<=popsize:
            subset=pop
            dissimilarity = quantify_dissimilarity(subset)
            energy = np.average(np.array([x.info["fitness"] for x in subset])) 
        else:  
            # Applies elitism: only the subset including the first structure are considered
            similarity_matrix = sim_matrix(pop)
            fitness_vector = np.array([x.info["fitness"] for x in pop])
            first = pop[0]
            pop = pop[1:]
            popsize=popsize-1
            
            indexes = [n for n in range(1,len(pop))]
            combins = itertools.combinations(indexes,popsize)
            subsets_data = dict()
            for subset in combins:
                # iterates through all subsets
                subset_list = [0]+list(subset)
                submatrix = similarity_matrix[np.ix_(subset)]
                length = len(submatrix)
                higher_tri = submatrix[np.triu_indices(length,k=1)]
                vector_length = len(higher_tri)
                summation = sum(higher_tri)
                dissimilarity = (vector_length - summation)/vector_length
                energy = np.average(fitness_vector[np.ix_(subset_list)])
                bypass = False
                if self.oldE != 0 and self.oldd != 0:
                    Erel = energy/self.oldE
                    drel = dissimilarity/self.oldd
                else:
                    Erel = abs(energy)
                    drel = dissimilarity
               #     bypass = True
                if Erel >=1:
                    Erelperceived = Erel**(fitness_preponderance)
                else:
                    Erelperceived = Erel
                # calculates the Q value
                product_rel = Erelperceived*drel
              #  worsening_index = False
               # if Erel < 0.9 or drel < 0.9:
                #    if not bypass:
                 #       worsening_index = True
              #  if not worsening_index:
                subsets_data[subset] = (product_rel,energy,dissimilarity)
                print(subset_list,energy,dissimilarity,Erel,drel,product_rel)
            if len(subsets_data) == 0:
                print("ERROR")
            # selects the best subset
            subset_numbers = max(subsets_data.items(),key=operator.itemgetter(1))[0]
            #subset = [indexes[i-1] for i in subset_numbers]
            energy = subsets_data[subset_numbers][1]
            dissimilarity = subsets_data[subset_numbers][2]
            pop = [first]+pop
            subset = [pop[0]]+[pop[i] for i in subset_numbers]
            return subset,dissimilarity,energy

    def print_params(self):
         return "Criterion class"
