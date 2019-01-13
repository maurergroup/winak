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
from datetime import datetime
from winak.SOAP_interface import compare
import pandas as pd

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



class PopulationEvaluator:
    """Performs local optimization on every structure in the population, and tags it with a fitness value. Returns the optimized structure population"""
    __metaclass__ = ABCMeta

    def __init__(self,EE,EEparameters):
        """subclasses must call this method. EEparameters must be a dictionary"""
        self.EE=NAMESPACE[EE](**EEparameters)
        
    @abstractmethod
    def EvaluatePopulation(self,pop):
        """subclasses must implement this method. Has to return a population of optimized and evaluated structures"""
        pass
    
    @abstractmethod
    def print_params(self):
        """subclasses must implement this method. Has to return a string containing
        alli the important parameters"""
        pass

class BasicPopEvaluator(PopulationEvaluator):
    def __init__(self,EE,EEparameters,sim_threshold=0.999):
        """ This class evaluates every structure according to its energy, eliminates identical structures employing SOAP similarity indexes, and returns the new evaluated population, a report and a numerical report """
        PopulationEvaluator.__init__(self,EE,EEparameters)
        self.similarity_threshold=sim_threshold
    def EvaluatePopulation(self,popreceived):
        """Evaluates every structure in the population: Calls the selected EE to perform local optimization and evaluate energy. Eliminates identical structures. Writes the fitness in info['fitness']"""
        pop = popreceived[:]
        EvaluatedPopulation = []
        successfulmat = 0
        successfulmut = 0
        report = ""
        #Isolates the new structures, putting the old ones directly in EvaluatedPopulation
        popofnews=pop[:]
        for stru in pop:
            oldone = False
            if hasattr(stru,"info"):
                if "New" in stru.info:
                    if stru.info["New"] == False:
                         oldone = True
            if oldone:
                EvaluatedPopulation.append(stru.copy())
                popofnews.remove(stru)
        pop = popofnews[:]
        detailed_report = "Splitting population in "+str(len(EvaluatedPopulation))+" old structures and "+str(len(pop))+" new structures."

        # optimizes all the new structures
        time0 = datetime.now()
        OptimizedPopulation = []
        failed_optimizations = []
        for stru in pop:
            failed = stru.copy()
            result = self.EE.get_energy(stru)
            if result is None:
                failed_optimizations.append(failed) 
            else:
                NewStructure = result[0]
                fitness = -result[1]
                NewStructure.info["fitness"] = fitness
                OptimizedPopulation.append(NewStructure.copy())
        
        detailed_report += "\n"+"Optimization failed for "+str(len(failed_optimizations))+" structures."
        counter = 1
        for structure in failed_optimizations:
            description ="\n"+ "%s:  = %s" %(str(counter),structure.get_chemical_formula())
            if "Origin" in structure.info:
                description += " %s, from previous generation [%s]"%(structure.info["Origin"],structure.info["Ascendance"])
            detailed_report += description
            counter += 1

        time1=datetime.now()
        # checks the optimized structures for duplicates
        detailed_report += "\n"+"\n"+"Eliminating duplicates:"
        OptimizedPopulation,elimination_report = self.__eliminate_duplicates(OptimizedPopulation,EvaluatedPopulation)
        detailed_report += elimination_report

        time2=datetime.now()
        # records the successfull optimization, and returns the final evaluated population
        for stru in OptimizedPopulation:
            if hasattr(stru,"info"):
                if "Origin" in stru.info:
                    if stru.info["Origin"] == "Child":
                         successfulmat += 1
                    elif stru.info["Origin"] == "Mutated":
                         successfulmut += 1
            EvaluatedPopulation.append(stru.copy())

        report += "\n"+"Successfully optimized new structures: "+str(successfulmut+successfulmat)+"\n"+"Generated by mating: "+str(successfulmat)+"\n"+"Generated by mutation: "+str(successfulmut)+"\n"+"Evaluation time: "+str(time2-time0)
        
        report += "\n"+"\n"+"__ Evaluated Structures __"
        counter = 1 
        for structure in EvaluatedPopulation:
             if hasattr(structure,"info"):
                if "New" in structure.info:
                    if structure.info["New"]:
                        description ="\n"+ "%s: Fitness = %s" %(str(counter),structure.info["fitness"])
                        description += " - %s, from previous generation [%s]"%(structure.info["Origin"],structure.info["Ascendance"])
                        report += description
                        counter += 1


        report += "\n"+"____________________________________"+"\n"
        report += "\n"+"Evaluation Details: "+"\n"
        report += "\n"+"Actual evaluation time: "+str(time1-time0)
        report += "\n"+"Structure comparison time: "+str(time2-time1)+"\n"
        report += detailed_report

        num_report=pd.DataFrame({'Successfully optimized new structures':(successfulmut+successfulmat),'Evaluation Time':(time1-time0),'Comparison Time':(time2-time1)},index=[2])
        return EvaluatedPopulation,report,num_report
    

    def __check_identity(self,a,b):
        """returns True if a and b are identical, and False if they are not"""
        #first, check stoichiometry
        if a.get_chemical_formula() != b.get_chemical_formula():
            return False,"Different Chemical Formula"
        
        report="\n"+"Energies: a = "+str(a.info["fitness"])+" b = "+str(b.info["fitness"])
        #second, check energy
        if abs(a.info["fitness"] - b.info["fitness"]) > 2:
            return False,"Different Energy"
        
        #third, check structure
        compare_result = compare(a,b)
        if compare_result < self.similarity_threshold:
            return False,"SOAP similarity value = "+str(compare_result)
                     
        report +="\n"+"Structure found equivalent to: "
        report += "\n"+ "%s - Fitness = %s" %(b.get_chemical_formula(),b.info["fitness"])
        if "Origin" in b.info:    
            report += " %s, from previous generation [%s]"%(b.info["Origin"],b.info["Ascendance"])
       
        report += "\n"+ "SOAP similarity value = "+str(compare_result)+"\n"
        return True,report
    
    def __eliminate_duplicates(self,NewPop,OldPop):
        """eliminates all duplicates within NewPop. Then, eliminates all the elements of NewPop that are already in OldPop. Returns the 'clean' NewPop."""
        report = ""
        # Eliminates all duplicates within NewPop
        to_remove=[]
        reports=[]
        for stru in NewPop:
            if stru not in to_remove:
                all_the_others = NewPop[:]
                all_the_others.remove(stru)
                for stru2 in all_the_others:
                    result,identity_report = self.__check_identity(stru2,stru)
                    if result:
                        to_remove.append(stru2)
                        reports.append(identity_report)
        report += "\n"+str(len(to_remove))+" structures identified as redundant and eliminated."+"\n"
        index_number = 0
        for structure in to_remove:
            description ="\n"+ "%s - Fitness = %s" %(structure.get_chemical_formula(),structure.info["fitness"])
            description += " %s, from previous generation [%s]"%(structure.info["Origin"],structure.info["Ascendance"])
            report += description
            report += reports[index_number]+"\n"
            index_number += 1
            try:
                NewPop.remove(structure)
            except:
                report +=  "FAILED REMOVAL"

        # Eliminates all the structures in NewPop that already appear in OldPop
        popcopy = NewPop[:]
        to_remove=[]
        reports = []
        for stru in popcopy:
            to_remove.append(stru)
            reports.append(identity_report)
            break
        report += "\n"+str(len(to_remove))+" structures identified as equivalent to old structures and eliminated."+"\n"
        
        index_number = 0
        for structure in to_remove:
            description ="\n"+ "%s - " %(structure.get_chemical_formula())
            
            if hasattr(stru,"info"):
                if "Origin" in stru.info:
                    description += " %s, from previous generation [%s]"%(structure.info["Origin"],structure.info["Ascendance"])
            report += description
            report += reports[index_number]
            index_number += 1
            NewPop.remove(structure)
            
        return NewPop,report

    def print_params(self):
        return "BasicPopEvaluator, employing evaluator "+self.EE.print_params()

NAMESPACE=locals()
