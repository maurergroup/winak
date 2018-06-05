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

    def __init__(self, pop,
                 EnergyEvaluator,
                 Displacer,
                 Criterion,
                 logfile='GS.log',
                 savegens=False,
                 break_limit=None,
                 break_limit_top=None):
        self.pop = pop
        self.logfile=logfile
        self.eneval=EnergyEvaluator
        self.displacer=Displacer
        self.crit=Criterion
        self.startT = datetime.now()
        self.log('STARTING Screening at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+' KK 2015 / FC 2018')
        self.log('Using the following Parameters and Classes:')
        self.log('EnergyEvaluator - '+self.eneval.print_params())
        self.log('Displacer - '+self.displacer.print_params())
        self.log('Criterion - '+self.crit.print_params())
        self.savegens=savegens #if True, saves a .traj file for every generation
        if break_limit == None:  #if these number of consecutive generations have no new structure, the cycle breaks
            self.break_limit = -1
        else:
            self.break_limit = break_limit     
            
        if break_limit_top == None:   #if the best structure remains the same for this number of consecutive generations, the cycle breaks
            self.brek_limit_top = -1
        else:
            self.break_limit_top = break_limit_top

    def run(self, gens):
        """Screen for defined number of generations. If break_limit is set, stops after n. break_limit consecutive unmodified generations """
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
            self.log("\n"+"\n"+"Generation "+str(gen+1))
            #Displacing = creation of the new candidates
            NewCandidates,report = self.displacer.evolve(self.pop)
            self.log(report)
            #EE = local optimization and tagging of the new structures
            EvaluatedPopulation,report = self.eneval.EvaluatePopulation(NewCandidates)
            self.log(report)
            #criterion = definition of the next generation
            newgen,report = self.crit.filter(EvaluatedPopulation)
            self.log(report)
            if self.pop == newgen:
                break_limit -= 1
            else:
                break_limit = self.break_limit
            if self.pop[0] == newgen[0]:
                break_limit_top -= 1
            else:
                break_limit_top = self.break_limit_top
            self.pop = newgen
            if self.savegens:
                if not os.path.exists("Generations"):
                    os.mkdir("Generations")
                os.chdir("Generations")
                write(str(gen+1)+"_generation.traj",self.pop)
                os.chdir("..")
        if os.path.exists("relax.traj"):
            os.remove("relax.traj")
        write("Results.traj",self.pop)
        self.endT = datetime.now()
        self.log('ENDING Screening at '+self.endT.strftime('%Y-%m-%d %H:%M:%S'))
        self.log('Time elapsed: '+str(self.endT-self.startT))            
       
    def log(self, msg):
        if self.logfile is not None:
            with open(self.logfile,'a') as fn:
                fn.write(msg+'\n')
                fn.flush()
