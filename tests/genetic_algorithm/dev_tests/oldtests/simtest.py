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
from ase.calculators.emt import EMT

import numpy as np

pop=[]
stru1 = Atoms('O2N2',positions=[(0,0,0),(2,2,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])
stru2 = Atoms('O2N2',positions=[(2,0,0),(0,2,0),(0,0,0),(2,2,0)],cell=[4,4,10],pbc=[1,1,0])
stru3 = Atoms('O2N2',positions=[(0,0,0),(2,2,0),(0.7,1.3,0),(1.3,0.7,0)],cell=[4,4,10],pbc=[1,1,0])
stru4 = Atoms('O2N2',positions=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)],cell=[4,4,10],pbc=[1,1,0])
stru5 = Atoms('O2N3',positions=[(0,0,0),(2,2,0),(2,0,0),(0,2,0),(1,1,0)],cell=[4,4,10],pbc=[1,1,0])
stru6 = Atoms('O2N2',positions=[(0,0,0),(1.9,1.9,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])
stru7 = Atoms('O2N2',positions=[(0,0,0),(1.8,1.8,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])

pop.append(stru1)
pop.append(stru2)
pop.append(stru3)
pop.append(stru4)
pop.append(stru5)
pop.append(stru6)
pop.append(stru7)

write('pop.traj',pop)
write('pop.xyz',pop)
result = Trajectory('pop.traj','r')
view(result)


