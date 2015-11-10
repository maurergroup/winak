from ase.all import *
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from winak.globaloptimization.betterhopping import BetterHopping
import numpy as np

adsorbate = '../testsystems/NH3_on_Ag100.traj'
molecule = read(adsorbate)

#NOTE: Define your own calculator here!
molecule.set_calculator(EMT())

f = FixAtoms(mask=[atom.symbol == 'Ag' for atom in molecule])


bh = BetterHopping(atoms=molecule,
                  temperature=100 * kB,
                  dr=1.0,
                  optimizer=BFGS,
                  fmax=2,
                  logfile='tt.log',
                  maxmoves=50,
                  movemode=1,
                  numdelocmodes=3,
                  adsorbmask=(8,12),
                  cell_scale=(0.5,0.5,0.1)
                  )
bh.run(20)
