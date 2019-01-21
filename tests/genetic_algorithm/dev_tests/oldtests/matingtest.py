from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import SinusoidalCut
from winak.screening.displacer import TestMating
import os
from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

###automatically deletes last execution's results
if os.path.exists("Results.traj"):
    os.remove("Results.traj")

#struA = read('ocus.traj',2)
#struB = read('obr.traj',2)


pop = []
   
materials=['Au','Cu']
      
for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info,pbc=[1,1,0])
    stru = stru.repeat((3,3,2))
    pop.append(stru.copy())
 
struA=pop[0]
struB=pop[1]


### fixing the bulk atoms. will be automatized in the population generator
#for atom in struA:
#    if (-2.5) < atom.z < 2.5:
#        atom.tag = 1

#for atom in struB:
#    if (-2.5) < atom.z < 2.5:
#        atom.tag = 1



Parameters = {"FixedElements":["Ru"],"NumberOfAttempts":5,"collision_threshold":0.5}
MatingOperator = SinusoidalCut(**Parameters) 

Children = MatingOperator.Mate(struA,struB)



ToVisualize = [struA,struB]+ Children


write('Results.traj',ToVisualize)







