from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from ase.constraints import FixAtoms
from winak.screening.displacer import FabioManager
from winak.screening.displacer import GC
from winak.screening.energyevaluator import FabioPopEvaluator
from winak.screening.criterion import FabioSelection
from winak.screening.ultimatescreener import GeneticScreener
import os

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np




materials=['Ni','Al','Au','Cu']



pop=[]
for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,2.5,0),(2.5,0,0),(2.5,2.5,0)],cell=[5,5,2.5],info=info,pbc=[1,1,0])
    stru = stru.repeat((3,3,3))
    for atom in stru:
        if atom.z < 2:
            atom.tag = 2
        elif atom.z > 2 and atom.z < 3:
            atom.tag = 1
    pop.append(stru.copy())


structure = pop[2]
print("structure ready")
constraint = FixAtoms(mask=[(atom.tag == 2 or atom.tag == 1) for atom in structure])
structure.set_constraint(constraint)
print("constraint set")
displacer = GC(prob = 0.0, adsorbate = 0, periodic = True,constrain = True)
print("displacer ready")
result = displacer.displace(structure)
print("displaced")
del(result.constraints)
constraint = FixAtoms(mask=[atom.tag == 2 for atom in result])
result.set_constraint(constraint)
view(result)
