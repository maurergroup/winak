
from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager
from winak.energyevaluator import FabioPopEvaluator


import numpy as np

pop = []
 
materials=['Ni','Co','Zn','Cu']


for el in materials:
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,3,0),(3,0,0),(3,3,0)],cell=    [6,6,3],info=info)
    stru = stru.repeat((3,3,2))
    pop.append(stru.copy())

write('pop.traj',pop)

population = Trajectory('pop.traj','r')

########## Controls ##########
EE = "potEE"
EEparameters = {"calc":"EMT()","opt":"BFGS","fmax"=1.0}
##############################


PopulationEvaluator = FabioPopEvaluator(EE,EEparameters)
evalpop = PopulationEvaluator.EvaluatePopulation(population)

for stru in evalpop:
    print(stru.info)
