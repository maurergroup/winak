import ase
from ase.io import read
from ase.visualize import view
import numpy as np

from ase.structure import molecule
from obcalc import OBForceField
from ase.optimize import BFGS

atoms = molecule('CO')

calc = OBForceField()
atoms.set_calculator(calc)
e = atoms.get_potential_energy()

dyn = BFGS(atoms)
dyn.run(0.01)


#a = read ('ammonia.gen')

