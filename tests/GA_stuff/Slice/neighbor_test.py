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
from ase.visualize import view

from ase.data import covalent_radii
from ase.data import atomic_numbers
from ase.neighborlist import NeighborList

import numpy as np

###automatically deletes last execution's results
if os.path.exists("Results.traj"):
    os.remove("Results.traj")


TestStru = Atoms('Ru3O3',[(0,0,0),(0,0,8),(0,0.8*2.92,8),(0,0.8*2.12,0),(0,0,4),(0,0.8*1.32,4)])

struA = read('ocus.traj',2)
#struB = read('obr.traj',2)
view(struA)
print("Ru:",covalent_radii[atomic_numbers["Ru"]])
print("O",covalent_radii[atomic_numbers["O"]])
#print([atom.symbol for atom in struA])

radii = [(0.8)*covalent_radii[atomic_numbers[atom.symbol]] for atom in TestStru]

lammazzatore = NeighborList(cutoffs=radii,self_interaction=False)

lammazzatore.build(TestStru)
print(lammazzatore.nneighbors)

view(TestStru)







