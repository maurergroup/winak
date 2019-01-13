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
import pandas as pd
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
                 num_logfile = 'numerical_log.xlsx',
                 savegens=False,
                 break_limit=None,
                 break_limit_top=None):
        """ This class runs the algorithm through a given number of generations, calling the displacement, evaluation and selection classes cyclically.
            Records the results on a .log file in human-readable form, and a condensed version of the results on a numerical_log file designed for plotting and data-visualization """
        self.logfile=logfile
        self.num_logfile = num_logfile
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
        
        parameters = {'Displacement Stepwidth':self.displacer.MutationManager.MutationOperator.stepwidth,
                      'Population size':self.crit.popsize,
                      'Fitness preponderance':self.crit.fitness_preponderance,
                      'xParameter':self.displacer.Xparameter}
        parameters = pd.DataFrame(parameters,index=['Parameters'])
        self.num_log(parameters)

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
            # checks for termination conditions
            self.gen_startT = datetime.now()
            if break_limit == 0:
                self.log("Break limit. Interrupting.")
                break
            if break_limit_top == 0:
                self.log("Break limit top. Interrupting.")
                break
            gentime = datetime.now()
            self.log("\n"+"______________________________________________________"+"\n"+"______________________________________________________"+"\n"+"\n"+"Generation "+str(gen+1)+"                       "+str(gentime.strftime('%Y-%m-%d %H:%M:%S'))+"\n"+"______________________________________________________"+"\n"+"______________________________________________________")
            num_log=pd.DataFrame(index=[gen+1])
            #Displacing = creation of the new candidates
            self.log("\n"+"____________DISPLACING____________")
            NewCandidates,report,num_report = self.displacer.evolve(pop)
            self.log(report)
            num_report.index=[(gen+1)]
            num_log = pd.concat([num_log,num_report],axis=1)
            #EE = local optimization and tagging of the new structures
            self.log("\n"+"____________EVALUATING____________")
            EvaluatedPopulation,report,num_report = self.eneval.EvaluatePopulation(NewCandidates)
            self.log(report)
            num_report.index=[(gen+1)]
            num_log = pd.concat([num_log,num_report],axis=1)
            #criterion = definition of the next generation          
            self.log("\n"+"____________SELECTING_____________")
            newgen,report,diversity,fitness,first_fitness,num_report = self.crit.filter(EvaluatedPopulation,mode="best_compromise_elite")
            self.log(report)
            # numerical logging
            num_report.index=[(gen+1)]
            num_log = pd.concat([num_log,num_report],axis=1)
          
            self.log("\n"+"Average fitness: "+str(fitness))
            self.log("Diversity: "+str(diversity))
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
            self.gen_endT = datetime.now()
            #records detailed times in numerical_log
            self.gen_time = self.gen_startT - self.gen_endT
            spurious_time = (self.gen_time - (num_log.loc[(gen+1),'Mating Time'] + num_log.loc[(gen+1),'Mutation Time'] + num_log.loc[(gen+1),'Comparison Time'] +num_log.loc[(gen+1),'Evaluation Time'] +num_log.loc[(gen+1),'Selection Time']))
            displacement_time = (num_log.loc[(gen+1),'Mating Time'] + num_log.loc[(gen+1),'Mutation Time'])
            soap_time = (num_log.loc[(gen+1),'Comparison Time'] + num_log.loc[(gen+1),'Selection Time'])
            last_data = pd.DataFrame({'Displacement Time':displacement_time,'SOAP Time':soap_time,'Spurious Time':spurious_time},index=[gen+1])
            num_log = pd.concat([num_log,last_data],axis=1)
            self.num_log(num_log)
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
        # initializes the generator
        generator = self.displacer
        self.log('Generator - '+generator.MutationManager.print_params())
        self.log('Criterion - '+self.crit.print_params())
        pop = []
        # adapts to single structure/population inputs
        if type(strus)==list:
            for stru in strus:
                pop.append(stru.copy())
        else:
            pop.append(strus.copy())
        # applies constraints to layer 2 of every atom
        for structure in pop:
            constraint = FixAtoms(mask=[atom.tag == 2 for atom in structure])
            structure.set_constraint(constraint)
        diversity = 0
        cycle_count = 1
        # launches the energy evaluation of the provided structures
        self.log("\n"+"_________INITIAL EVALUATION__________")
        pop, report, num_log = self.eneval.EvaluatePopulation(pop)
        #numerical logging
        num_log.index=['gen 0']
        self.num_log(num_log)
        # checks for empty input populations
        if len(pop) == 0:
            self.log('\n'+"EXECUTION TERMINATED - Initial evaluation has failed, or an invalid input was selected"+'\n')
        # starts generating
        while len(pop) < popsize or diversity < minimum_diversity:    
            self.gen_startT = datetime.now()
            num_log = pd.DataFrame(index=[str('gen'+str(cycle_count))]) 
            # displacing
            self.log("\n"+"_____________________________________________________________"+"\n")
            self.log("\n"+"Producing initial population: cycle n."+str(cycle_count))
            self.log("\n"+"_____________________________________________________________"+"\n")
            self.log("\n"+"____________DISPLACING____________")
            pop, report, num_report = generator.evolve(pop,Xpar=0)
            self.log(report) 
            num_report.index=[str('gen'+str(cycle_count))]
            num_log = pd.concat([num_log,num_report],axis=1)
            # evaluating
            self.log("\n"+"____________EVALUATING____________")
            pop, report,num_report = self.eneval.EvaluatePopulation(pop)
            num_report.index=[str('gen'+str(cycle_count))]
            num_log = pd.concat([num_log,num_report],axis=1)
            self.log(report) 
            # selecting
            self.log("\n"+"____________SELECTING_____________")
            pop, report, diversity, fitness, first_fitness, num_report = self.crit.filter(pop,mode="best_diversity")
            self.log(report)
            num_report.index=[str('gen'+str(cycle_count))]
            num_log = pd.concat([num_log,num_report],axis=1)
            self.log("\n"+"Population size: "+str(len(pop))+" / "+str(popsize))
            self.log("\n"+"Average fitness: "+str(fitness))
            self.log("Diversity: "+str(diversity)+" / "+str(minimum_diversity))
            self.gen_endT = datetime.now()i
            # detailed execution time logging
            self.gen_time = self.gen_startT - self.gen_endT
            spurious_time = (self.gen_time - (num_log.loc[str('gen'+str(cycle_count)),'Mating Time'] + num_log.loc[str('gen'+str(cycle_count)),'Mutation Time'] + num_log.loc[str('gen'+str(cycle_count)),'Comparison Time'] +num_log.loc[str('gen'+str(cycle_count)),'Evaluation Time'] +num_log.loc[str('gen'+str(cycle_count)),'Selection Time']))
            displacement_time = (num_log.loc[str('gen'+str(cycle_count)),'Mating Time'] + num_log.loc[str('gen'+str(cycle_count)),'Mutation Time'])
            soap_time = (num_log.loc[str('gen'+str(cycle_count)),'Comparison Time'] + num_log.loc[str('gen'+str(cycle_count)),'Selection Time'])
            last_data = pd.DataFrame({'Displacement Time':displacement_time,'SOAP Time':soap_time,'Spurious Time':spurious_time},index=[str('gen'+str(cycle_count))])
            num_log = pd.concat([num_log,last_data],axis=1)
            self.num_log(num_log)
            cycle_count += 1
        # resets the attributes of all structures before the execution of the Evolve function
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
    def num_log(self,dfr):
        if self.num_logfile is not None:
            try:
                previous = pd.read_excel(self.num_logfile)
            except:
                previous = pd.DataFrame()
            new = pd.concat([previous,dfr],sort=False)
            writer = pd.ExcelWriter(self.num_logfile)
            new.to_excel(writer,'Sheet1')
            writer.save()

