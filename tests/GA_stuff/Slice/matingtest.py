from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import SinusoidalCut
from winak.screening.displacer import TestMating

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

struA = read('ocus.traj',2)
struB = read('obr.traj',2)

### fixing the bulk atoms. will be automatized in the population generator
for atom in struA:
    if (-2.5) < atom.z < 2.5:
        atom.tag = 1

for atom in struB:
    if (-2.5) < atom.z < 2.5:
        atom.tag = 1



Parameters = {"FixedElements":["Ru"]}
MatingOperator = SinusoidalCut(**Parameters) 

Children = MatingOperator.Mate(struA,struB)



ToVisualize = [struA,struB]+ Children


write('Results.traj',ToVisualize)








