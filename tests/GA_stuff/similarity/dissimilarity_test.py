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
from winak.SOAP_interface import quantify_dissimilarity 
import os
import shutil

from ase.optimize import BFGS
from ase.calculators.emt import EMT

import numpy as np

pop = []
for cont in range(4):
    stru = read("/data/guest20/Executables/winak/tests/GA_stuff/Generations/5_generation.traj",cont)
    pop.append(stru)

result = quantify_dissimilarity(pop)
print(result)

