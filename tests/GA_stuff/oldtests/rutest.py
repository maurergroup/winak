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
from ase.calculators.morse import MorsePotential

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
EEparameters = {"calc":MorsePotential(),"opt":BFGS,"fmax":1.0,"optlog":'optimization.log'}

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

struA = read('oldtests/obr.traj',2)
struB = read('oldtests/ocus.traj',2)
strus = [struA,struB]

starting_population = Screener.generate(strus,4)
write("starting_population.traj",starting_population)

Screener.run(starting_population,3)



