from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager

import numpy as np

pop = [] 

materials=['Ni','Co','Zn','Cu']


for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,3,0),(3,0,0),(3,3,0)],cell=[6,6,3],info=info)
    stru = stru.repeat((3,3,2))
    pop.append(stru.copy())



################ Controls ################

Xparameter = 2 

MatingManager = "FabioMating"
MutationManager = "FabioMutation"

MatingOperator = "TestMating"
MatingOperatorParameters = {}

MutationOperator = "TestMutationOperator"
MutationOperatorParameters = {}

##########################################



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

########## set 
GeneticDisplacer = FabioManager(**Parameters)

for cont in range(50):
    population = GeneticDisplacer.evolve(pop)
    listpop = []
    for structure in population:
        listpop.append(structure.copy())

    np.random.shuffle(listpop)
    
    pop = listpop[:4]
    print("POPULATION:")
    for struc in pop:
        print(struc.get_chemical_formula())

    print("Generation",cont,"completed")

write("GAresult.traj",pop)

GAresult = Trajectory('GAresult.traj','r')
view(GAresult)



