import ase
from ase.io import read
from ase.visualize import view
import numpy as np

from ase.structure import molecule
#from obcalc import OBForceField
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from winak.optimize import DIopt

atoms =read('../testsystems/Cucluster.xyz')
#atoms =read('../testsystems/peptide.xyz')
atoms.rattle()

calc = EMT()
atoms.set_calculator(calc)
e = atoms.get_potential_energy()

dyn = BFGS(atoms)
dyn.run(0.1)

# view(atoms)

raw_input('press enter!')

atoms =read('../testsystems/Cucluster.xyz')
#atoms =read('../testsystems/clethen.xyz')
atoms.rattle()

#calc = OBForceField()
calc = EMT()
atoms.set_calculator(calc)

dyn = DIopt(atoms)
dyn.run(0.1)

view(atoms)

