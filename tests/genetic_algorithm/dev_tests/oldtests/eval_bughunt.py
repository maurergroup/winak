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
from ase.calculators.morse import MorsePotential

import numpy as np


############### Evaluation ###################
EE = "potEE"
EEparameters = {"calc":MorsePotential(),"opt":BFGS,"fmax":1.0,"optlog":'optimization.log'}

 
PopulationEvaluator = FabioPopEvaluator(EE,EEparameters)




struA = read('oldtests/obr.traj',2)

pop = [struA]
evaluated = PopulationEvaluator.EvaluatePopulation(pop)
print(evaluated)
