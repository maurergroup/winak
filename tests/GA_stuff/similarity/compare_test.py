from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager
from winak.screening.energyevaluator import FabioPopEvaluator
from winak.screening.criterion import FabioSelection
from winak.screening.ultimatescreener import GeneticScreener
from winak.SOAP_interface import compare
import os
import shutil

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

pair_1=[]
pair_2=[]
pair_3=[]
pair_4=[]

stru1 = Atoms('O2N2',positions=[(0,0,0),(2,2,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])
stru2 = Atoms('O2N2',positions=[(0,0,0),(1.9,1.9,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])

stru3 = Atoms('O2N2',positions=[(0,0,0),(2,2,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])
stru4 = Atoms('O2N2',positions=[(0,0,0),(1.7,1.7,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])

stru5 = Atoms('O2N2',positions=[(0,0,0),(2,2,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])
stru6 = Atoms('O2N2',positions=[(0,0,0),(1.2,1.2,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])

stru7 = Atoms('O2N2',positions=[(0,0,0),(2,2,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])
stru8 = Atoms('O2N2',positions=[(0,0,0),(0.8,0.8,0),(2,0,0),(0,2,0)],cell=[4,4,10],pbc=[1,1,0])

pair_1.append(stru1)
pair_1.append(stru2)
pair_2.append(stru3)
pair_2.append(stru4)
pair_3.append(stru5)
pair_3.append(stru6)
pair_4.append(stru7)
pair_4.append(stru8)

write('pair_1.traj',pair_1)

write('pair_2.traj',pair_2)

write('pair_3.traj',pair_3)

write('pair_4.traj',pair_4)

compare1 = compare(stru1,stru2)
compare2 = compare(stru3,stru4)
compare3 = compare(stru5,stru6)
compare4 = compare(stru7,stru8)

print("compare1 = "+str(compare1))
print("compare2 = "+str(compare2))
print("compare3 = "+str(compare3))
print("compare4 = "+str(compare4))



