import ase
from ase.io import read
from ase.visualize import view
import numpy as np

from ase.structure import molecule
from obcalc import OBForceField
from ase.optimize import BFGS

from winak.optimize import DIopt

atoms =read('../testsystems/clethen.xyz')

calc = OBForceField()
atoms.set_calculator(calc)
e = atoms.get_potential_energy()

dyn = BFGS(atoms)
dyn.run(0.01)

view(atoms)

raw_input('press enter!')

atoms =read('../testsystems/clethen.xyz')

dyn = DIopt(atoms)
dyn.run(0.01)

view(atoms)

