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


pop = [] 

materials=['Ni','Al','Au','Cu']


for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info,pbc=[1,1,0])
    stru = stru.repeat((3,3,2))
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
MutationOperatorParameters = {"prob":0.0,"periodic":True,"constrain":True,"GA_mode":True}

##############################################
############### Evaluation ###################
EE = "potEE"
EEparameters = {"calc":EMT(),"opt":BFGS,"fmax":1.0,"optlog":None}

##############################################
############### Criterion ####################
popsize = 8    

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
Criterion = FabioSelection(popsize)


Screener = GeneticScreener(PopulationEvaluator,GeneticDisplacer,Criterion,savegens=True, break_limit=break_limit, break_limit_top = break_limit_top)
Screener.run(pop,3)



