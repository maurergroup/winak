5
from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager
from winak.screening.energyevaluator import FabioPopEvaluator

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

pop = []
 
materials=['Ni','Al','Au','Cu']


for el in materials:
    info=dict()
    info["fitness"]=None
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info)
    stru = stru.repeat((3,3,2))
    pop.append(stru.copy())

write('pop.traj',pop)

population = Trajectory('pop.traj','r')

#stru = population[2]
#stru.set_calculator(EMT())
#opt = BFGS(stru,trajectory="testBFGS.traj")

#opt.run(fmax = 1.0)
 ########## Controls ##########
EE = "potEE"
EEparameters = {"calc":EMT(),"opt":BFGS,"fmax":1.0}
##############################


PopulationEvaluator = FabioPopEvaluator(EE,EEparameters)
evalpop = PopulationEvaluator.EvaluatePopulation(population)

for stru in evalpop:
    print(stru.info)
