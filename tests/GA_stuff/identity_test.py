from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager
from winak.screening.energyevaluator import FabioPopEvaluator
from winak.screening.criterion import FabioSelection
from winak.screening.ultimatescreener import GeneticScreener


from ase.optimize import BFGS
from ase.calculators.emt import EMT

from ase.ga.standard_comparators import InteratomicDistanceComparator as comp
import numpy as np

pop = [] 

materials=['Ni','Al','Au','Cu']


for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info)
    stru = stru.repeat((3,3,2))
    pop.append(stru.copy())
for stru in pop:
    stru.calc=(EMT())

comp=comp()
simil = comp.looks_like(pop[0],pop[1])


print(simil)












# ################ Controls ####################
# ################ Displacement ################
# Xparameter = 1 
# 
# MatingManager = "FabioMating"
# MutationManager = "FabioMutation"
# 
# MatingOperator = "TestMating"
# MatingOperatorParameters = {}
# 
# MutationOperator = "TestMutationOperator"
# MutationOperatorParameters = {}
# 
# ##############################################
# ############### Evaluation ###################
# EE = "potEE"
# EEparameters = {"calc":EMT(),"opt":BFGS,"fmax":1.0,"optlog":None}
# 
# ##############################################
# ############### Criterion ####################
# popsize = 4
# 
# ##############################################
# ############### Screening ####################
# break_limit = 30
# break_limit_top = 100
# 
# ##############################################
# 
# MatingParameters = {"MatingOperator":MatingOperator,
#                     "MatingOperatorParameters":MatingOperatorParameters
#                    } 
# MutationParameters = {"MutationOperator":MutationOperator,
#                       "MutationOperatorParameters":MutationOperatorParameters
#                      }
# DisplacementParameters = {"MatingManager":MatingManager,
#                           "MatingParameters":MatingParameters,
#                           "MutationManager":MutationManager,
#                           "MutationParameters":MutationParameters,
#                           "Xparameter":Xparameter
#                           }
# 
# 
#  
# GeneticDisplacer = FabioManager(**DisplacementParameters)
# PopulationEvaluator = FabioPopEvaluator(EE,EEparameters)
# Criterion = FabioSelection(popsize)
# 
# 
# Screener = GeneticScreener(pop,PopulationEvaluator,GeneticDisplacer,Criterion,savegens=False, break_limit=break_limit, break_limit_top = break_limit_top)
# Screener.run(500)
# 
# 
# 
