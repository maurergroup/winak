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
import shutil

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

#### deletes the results of the last execution
if os.path.exists("starting_population.traj"):
    os.remove("starting_population.traj")

if os.path.exists("Results.traj"):
    os.remove("Results.traj")

if os.path.exists("GS.log"):
    os.remove("GS.log")
 
if os.path.exists("Generations"):
    shutil.rmtree("Generations",ignore_errors=True)


info=dict()
info['fitness']=np.random.rand()
stru = Atoms('Au2Cu2',positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info,pbc=[1,1,0])
stru = stru.repeat((3,3,6))
stru.set_cell([15,15,30])

### atom.tag:  
### 0 = free atom
### 1 = atom fixed in crossover and mutation BUT free to relax
### 2 = atom fixed in crossover and mutation AND fixed in relaxations

for atom in stru:
    if 3 < atom.z < 8:
        atom.tag = 1
    elif atom.z < 3:
        atom.tag = 2
    else:
        atom.tag = 0

################ Controls ####################
################ Displacement ################
Xparameter = 2 

MatingManager = "FabioMating"
MutationManager = "FabioMutation"

MatingOperator = "SinusoidalCut"   
MatingOperatorParameters = {"FixedElements":["Ru"],"NumberOfAttempts":10,"collision_threshold":0.5}  ####collision_threshold should be chosen analyzing the behavior of the optimizer

MutationOperator = "GC"
MutationOperatorParameters = {"prob":0.5,"periodic":True}

##############################################
############### Evaluation ###################
EE = "potEE"
EEparameters = {"calc":EMT(),"opt":BFGS,"fmax":1.0,"optlog":None}

##############################################
############### Criterion ####################
popsize = 4

##############################################
############### Screening ####################
break_limit = 30
break_limit_top = 100

##############################################

MatingParameters = {"MatingOperator":MatingOperator,
                    "MatingOperatorParameters":MatingOperatorParameters
                   } 
MutationParameters = {"MutationOperator":MutationOperator,
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
Criterion = FabioSelection(popsize)


Screener = GeneticScreener(PopulationEvaluator,GeneticDisplacer,Criterion,savegens=True, break_limit=break_limit, break_limit_top = break_limit_top)


starting_population = Screener.generate(stru,popsize)
write("starting_population.traj",starting_population)

Screener.run(starting_population,3)



