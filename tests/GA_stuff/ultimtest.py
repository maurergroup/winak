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
from ase.calculators.morse import MorsePotential

from ase.calculators.emt import EMT

import numpy as np

#### deletes the results of the last execution
if os.path.exists("Results.traj"):
    os.remove("Results.traj")
if os.path.exists("GS.log"):
    os.remove("GS.log")



struA = read('obr.traj',2)
struB = read('ocus.traj',2)

pop = [struA,struB]


#### set constraints.
#### 1 = not involved in mutations and matings.
#### 2 = not involved in mutation and matings, and fixed in relaxations.
for stru in pop:
    for atom in stru:
        if atom.z < 3 and atom.z > -3:
            atom.tag = 1
        else:
            atom.tag = 0
        if atom.z < 2 and atom.z > -2:
            atom.tag = 2




################ Controls ####################
################ Displacement ################
Xparameter = 4 

MatingManager = "FabioMating"
MutationManager = "FabioMutation"

MatingOperator = "SinusoidalCut"   
MatingOperatorParameters = {"FixedElements":["Ru"],"NumberOfAttempts":50,"collision_threshold":0.001}  ####collision_threshold should be chosen analyzing the behavior of the optimizer

MutationOperator = "GC"
FixedElements_GC = ["Ru"]
MutationOperatorParameters = {"prob":0.2,"stepwidth":2.5,"periodic":True,"constrain":True,"GA_mode":True}

##############################################
############### Evaluation ###################
EE = "potEE"
EEparameters = {"calc":MorsePotential(),"opt":BFGS,"fmax":1.0,"optlog":None}

##############################################
############### Criterion ####################
popsize = 8    

##############################################
############### Screening ####################
break_limit = 30
break_limit_top = 100
savegens = False
number_of_generations = 5 
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
Criterion = FabioSelection(popsize)


Screener = GeneticScreener(PopulationEvaluator,GeneticDisplacer,Criterion,savegens=savegens, break_limit=break_limit, break_limit_top = break_limit_top)
starting_population = Screener.generate(pop,popsize)
Screener.run(starting_population,number_of_generations)


