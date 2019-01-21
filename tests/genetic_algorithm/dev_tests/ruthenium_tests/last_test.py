### complete script, except for DFTB implementation ###

from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager
from winak.screening.energyevaluator import FabioPopEvaluator
from winak.screening.criterion import FabioSelection
from winak.screening.ultimatescreener import GeneticScreener
import os

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

#### deletes the results of the last execution
if os.path.exists("Results.traj"):
    os.remove("Results.traj")
if os.path.exists("GS.log"):
    os.remove("GS.log")
if os.path.exists("numerical_log.xlsx"):
    os.remove("numerical_log.xlsx")

pop = [] 

materials=['Ni','Al','Au','Cu']


for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info,pbc=[1,1,0])
    stru = stru.repeat((3,3,3))
    for atom in stru:
        if atom.z < 2:
            atom.tag = 2
        elif atom.z > 2 and atom.z < 3:
            atom.tag = 1
    pop.append(stru.copy())



################ Controls ####################
################ Displacement ################
Xparameter = 2 

MatingManager = "FabioMating"
MutationManager = "FabioMutation"

MatingOperator = "SinusoidalCut"   
MatingOperatorParameters = {"FixedElements":["Ru"],"NumberOfAttempts":10,"collision_threshold":0.5}  ####collision_threshold should be chosen analyzing the behavior of the optimizer

MutationOperator = "GC"
FixedElements_GC = ["Ru"]
MutationOperatorParameters = {"prob":0.2,"periodic":True,"constrain":True,"GA_mode":True}

##############################################
############### Evaluation ###################
EE = "potEE"
EEparameters = {"calc":EMT(),"opt":BFGS,"fmax":1.0,"optlog":None}

##############################################
############### Criterion ####################
popsize = 8    
fitness_preponderance = 1
##############################################
############### Screening ####################
break_limit = 30
break_limit_top = 100

##############################################

MatingParameters = {"MatingOperator":MatingOperator,
                    "MatingOperatorParameters":MatingOperatorParameters
                   } 
MutationParameters = {"MutationOperator":MutationOperator,
                      "FixedElements_GC":FixedElements_GC,
                      "MutationOperatorParameters":MutationOperatorParameters
                     }
DisplacementParameters = {"MatingManager":MatingManager,
                          "MatingParameters":MatingParameters,
                          "MutationManager":MutationManager,
                          "MutationParameters":MutationParameters,
                          "Xparameter":Xparameter
                         }


 
GeneticDisplacer = FabioManager(**DisplacementParameters)
PopulationEvaluator = FabioPopEvaluator(EE,EEparameters)
Criterion = FabioSelection(popsize,fitness_preponderance)


Screener = GeneticScreener(PopulationEvaluator,GeneticDisplacer,Criterion,savegens=True, break_limit=break_limit, break_limit_top = break_limit_top)
starting_population = Screener.generate(pop,popsize)
Screener.run(starting_population,2)

os.system('echo | mutt -a numerical_log.xlsx -- fabio.calcinelli@gmail.com')
