#winak.screening.ultimatescreener
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

import numpy as np
from datetime import datetime
from ase.io.trajectory import Trajectory
from collections import Counter
from winak.screening.composition import Stoichiometry
try:
    from ase.build.tools import sort
except:
    from ase.utils.geometry import sort
import os
from ase.io import write
from ase.constraints import FixAtoms

class UltimateScreener:
    """UltimateScreener

    by Konstantin Krautgasser, November 2015
    """

    def __init__(self, atoms,
                 EnergyEvaluator,
                 Displacer,
                 Criterion,
                 trajectory='minima.traj',
                 logfile='tt.log',
                 savetrials=True):
        self.atoms=atoms
        self.logfile=logfile
        self.eneval=EnergyEvaluator
        self.displacer=Displacer
        self.crit=Criterion
        #self.traj=Trajectory(trajectory,'w')
        ### set initial trajectory with composition
        self.comp=Stoichiometry()
        self.comp.get(atoms)
        self.traj=self.comp.make_traj()
        ###
        self.startT = datetime.now()
        self.log('STARTING Screening at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+' KK 2015')
        self.log('Using the following Parameters and Classes:')
        self.log('EnergyEvaluator - '+self.eneval.print_params())
        self.log('Displacer - '+self.displacer.print_params())
        self.log('Criterion - '+self.crit.print_params())
        self.savetrials=savetrials ## if True, store all trial moves in folder 'trial'

    def run(self, steps):
        """Screen for defined number of steps."""
        tmp = self.eneval.get_energy(self.atoms.copy())
        if tmp is None:
            self.log('Initial Energy Evaluation Failed. Please check your calculator!')
        else:
            tmp[0].info={'accepted':True}
            self.current=tmp[0];self.Emin=tmp[1]
            self.crit.evaluate(tmp[0].copy(),tmp[1])
            self.traj.write(self.current)
            self.log('Initial Energy Evaluation done. Note that this is structure 0 in your trajectory. Energy = %f, Stoichiometry = %s' %(tmp[1],self.comp.stoich))
        self.fallbackatoms=self.current.copy()
        self.fallbackstep=-1
        if self.savetrials:
            os.system('mkdir -p trial') 
	
        for step in range(steps):    
            """I strictly use copies here, so nothing can be overwritten in a subclass.
            A step is tried 10 times, if it keeps failing, the structure is reset.
            If it still fails, abort, since something is clearly wrong."""
            tmp=None
            tries=0
            reset=False  
            failed=False          
            while tmp is None:
                try:
                    tmp=self.displacer.displace(self.current.copy())
                except:
                    tmp=None
                tries+=1
                if tmp is None:
                    self.log('Error while displacing, retrying')
                else:
                    if self.savetrials:
                        tmp.write('trial/trial'+str(step+1)+'.xyz')
                    tmp=self.eneval.get_energy(tmp.copy())
                    if tmp is None:
                        self.log('Error while evaluating energy, retrying')
                if tries>10 and not reset:
                    self.log('Repeated Error during current step, rolling back to step %d' %self.fallbackstep)
                    self.current=self.fallbackatoms.copy()
                    tries=0
                    reset=True
                if tries>10 and reset:
                    failed=True
                    break
            if failed:
                self.log('ABORTING. COULD NOT PERFORM STEP.')
                break
            
            """change trajectory if stoichiometry has changed"""
            comp=Stoichiometry()
            oldtraj=self.traj
            if comp.has_changed(tmp[0],self.current):
                newtraj = comp.make_traj()
                self.traj=newtraj
            
            """setting backup"""
            accepted=self.crit.evaluate(tmp[0].copy(),tmp[1])
            if accepted:
                tmp[0].info={'accepted':True}
            else:
                tmp[0].info={'accepted':False}
            self.traj.write(tmp[0])
            
            acc='not'
            if accepted:       
                self.current=tmp[0].copy()
                acc=''
                self.fallbackatoms=tmp[0].copy()
                self.fallbackstep=step
            else:
                self.traj=oldtraj
            self.log('%s - step %d done, %s accepted, Energy = %f, Stoichiometry = %s '%(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),step+1,acc,tmp[1], comp.stoich))
        self.endT = datetime.now()
        self.log('ENDING Screening at '+self.endT.strftime('%Y-%m-%d %H:%M:%S'))
        self.log('Time elapsed: '+str(self.endT-self.startT))
            
       
    def log(self, msg):
        if self.logfile is not None:
            with open(self.logfile,'a') as fn:
                fn.write(msg+'\n')
                fn.flush()



class GeneticScreener:
    """GeneticScreener
    built on UltimateScreener
    by Konstantin Krautgasser, November 2015
    """

    def __init__(self,
                 EnergyEvaluator,
                 Displacer,
                 Criterion,
                 logfile='GS.log',
                 savegens=False,
                 break_limit=None,
                 break_limit_top=None):
        self.logfile=logfile
        self.eneval=EnergyEvaluator
        self.displacer=Displacer
        self.crit=Criterion
        self.savegens=savegens #if True, saves a .traj file for every generation
        if break_limit == None:  #if these number of consecutive generations have no new structure, the cycle breaks
            self.break_limit = -1
        else:
            self.break_limit = break_limit     
            
        if break_limit_top == None:   #if the best structure remains the same for this number of consecutive generations, the cycle breaks
            self.break_limit_top = -1
        else:
            self.break_limit_top = break_limit_top

    def run(self, pop, gens):
        """Screen for defined number of generations. If break_limit is set, stops after n. break_limit consecutive unmodified generations. If break_limit_top is set, stops after n. tbreak_limit_top consecutive generations in which the best structure is unmodified """
        
        self.startT = datetime.now()
        self.log('STARTING Screening at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+' KK 2015 / FC 2018')
        self.log('Using the following Parameters and Classes:')
        self.log('EnergyEvaluator - '+self.eneval.print_params())
        self.log('Displacer - '+self.displacer.print_params())
        self.log('Criterion - '+self.crit.print_params())
        
        gens = int(gens)
        break_limit = self.break_limit
        break_limit_top = self.break_limit_top
        for gen in range(gens):
            if break_limit == 0:
                self.log("Break limit. Interrupting.")
                break
            if break_limit_top == 0:
                self.log("Break limit top. Interrupting.")
                break
            gentime = datetime.now()
            self.log("\n"+"______________________________________________________"+"\n"+"______________________________________________________"+"\n"+"\n"+"Generation "+str(gen+1)+"                       "+str(gentime.strftime('%Y-%m-%d %H:%M:%S'))+"\n"+"______________________________________________________"+"\n"+"______________________________________________________")
            #Displacing = creation of the new candidates
            self.log("\n"+"____________DISPLACING____________")
            NewCandidates,report = self.displacer.evolve(pop)
            self.log(report)
            #EE = local optimization and tagging of the new structures
            self.log("\n"+"____________EVALUATING____________")
            EvaluatedPopulation,report = self.eneval.EvaluatePopulation(NewCandidates)
            self.log(report)
            #criterion = definition of the next generation          
            self.log("\n"+"____________SELECTING_____________")
            newgen,report,diversity,fitness = self.crit.filter(EvaluatedPopulation,mode="best_compromise_elite")
            self.log(report)
            if pop == newgen:
                break_limit -= 1
            else:
                break_limit = self.break_limit
            if pop[0] == newgen[0]:
                break_limit_top -= 1
            else:
                break_limit_top = self.break_limit_top
            pop = newgen
            if self.savegens:
                if not os.path.exists("Generations"):
                    os.mkdir("Generations")
                os.chdir("Generations")
                write(str(gen+1)+"_generation.traj",pop)
                os.chdir("..")
        if os.path.exists("relax.traj"):
            os.remove("relax.traj")
        write("Results.traj",pop)
        self.endT = datetime.now()
        self.log('ENDING Screening at '+self.endT.strftime('%Y-%m-%d %H:%M:%S'))
        self.log('Time elapsed: '+str(self.endT-self.startT))            
           

    def generate(self,strus,popsize,minimum_diversity=0):
        """Generates a population of popsize structures, starting from a single input structure or from a smaller population. If a minimum_diversity value is selected, new structures will be generated until a population is created that has such a diversity value (as expressed in the criterion method associated to the screener)"""
        
        self.startT = datetime.now()
        self.log('STARTING Generating at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+' KK 2015 / FC 2018')
        self.log('Using the following Parameters and Classes:')
        self.log('EnergyEvaluator - '+self.eneval.print_params())

        generator = self.displacer

        self.log('Generator - '+generator.MutationManager.print_params())
        self.log('Criterion - '+self.crit.print_params())
        pop = []
        if type(strus)==list:
            for stru in strus:
                pop.append(stru.copy())

        else:
            pop.append(strus.copy())

        for structure in pop:
            constraint = FixAtoms(mask=[atom.tag == 2 for atom in structure])
            structure.set_constraint(constraint)
        diversity = 0
        cycle_count = 1

        self.log("\n"+"_________INITIAL EVALUATION__________")
        pop, report = self.eneval.EvaluatePopulation(pop)
        if len(pop) == 0:
            self.log('\n'+"EXECUTION TERMINATED - Initial evaluation has failed, or an invalid input was selected"+'\n')
        while len(pop) < popsize or diversity < minimum_diversity:    
            self.log("\n"+"_____________________________________________________________"+"\n")
            self.log("\n"+"Producing initial population: cycle n."+str(cycle_count))
            self.log("\n"+"_____________________________________________________________"+"\n")
            self.log("\n"+"____________DISPLACING____________")
            pop, report = generator.evolve(pop,Xpar=0)
            self.log(report) 
            self.log("\n"+"____________EVALUATING____________")
            pop, report = self.eneval.EvaluatePopulation(pop)
            self.log(report) 
            self.log("\n"+"____________SELECTING_____________")
            pop, report, diversity, fitness = self.crit.filter(pop,mode="best_diversity")
            self.log(report)
            self.log("\n"+"Population size: "+str(len(pop))+" / "+str(popsize))
            self.log("\n"+"Average fitness: "+str(fitness))
            self.log("Diversity: "+str(diversity)+" / "+str(minimum_diversity))
            cycle_count += 1
        for stru in pop:
            stru.info.pop("Ascendance",None)
            stru.info.pop("Origin",None)
            stru.info.pop("New",None)
        if os.path.exists("relax.traj"):
            os.remove("relax.traj")
        self.endT = datetime.now()
        self.log("\n")
        self.log('Initial Population successfully generated at '+self.endT.strftime('%Y-%m-%d %H:%M:%S'))
        self.log('Time elapsed: '+str(self.endT-self.startT)+"\n"+"\n")            
        return pop


    def log(self, msg):
        if self.logfile is not None:
            with open(self.logfile,'a') as fn:
                fn.write(msg+'\n')
                fn.flush()


