from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager
from winak.screening.energyevaluator import FabioPopEvaluator
from winak.screening.criterion import FabioSelection

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

pop = [] 

materials=['Ni','Al','Au','Cu']


for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info)
    stru = stru.repeat((3,3,2))
    pop.append(stru.copy())



################ Controls ################
################ Displacement ###############
Xparameter = 2 

MatingManager = "FabioMating"
MutationManager = "FabioMutation"

MatingOperator = "TestMating"
MatingOperatorParameters = {}

MutationOperator = "TestMutationOperator"
MutationOperatorParameters = {}

#############################################
############### Evaluation #################
EE = "potEE"
EEparameters = {"calc":EMT(),"opt":BFGS,"fmax":1.0}

##############################################
############### Criterion ####################
popsize = 4

##############################################


MatingParameters = {"MatingOperator":MatingOperator,
                    "MatingOperatorParameters":MatingOperatorParameters
                   } 
MutationParameters = {"MutationOperator":MutationOperator,
                      "MutationOperatorParameters":MutationOperatorParameters
                     }
Parameters = {"MatingManager":MatingManager,
              "MatingParameters":MatingParameters,
              "MutationManager":MutationManager,
              "MutationParameters":MutationParameters,
              "Xparameter":Xparameter
             }

 
GeneticDisplacer = FabioManager(**Parameters)
PopulationEvaluator = FabioPopEvaluator(EE,EEparameters)
Criterion = FabioSelection()



###TO BE IMPLEMENTED: GENETIC SCREENER
for cont in range(10):
    pop = GeneticDisplacer.evolve(pop)
    print("Displacement complete. Population:",pop)
    pop = PopulationEvaluator.EvaluatePopulation(pop)
    print("Evaluation complete. Population:",pop)
    pop = Criterion.filter(pop,popsize)
    print("Selection complete. Population:",pop)
    
    for struc in pop:
        print(struc.get_chemical_formula(),struc.info)

    print("Generation",cont,"completed")

write("GAresult.traj",pop)

GAresult = Trajectory('GAresult.traj','r')
view(GAresult)



