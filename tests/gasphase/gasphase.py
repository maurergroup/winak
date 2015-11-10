from ase.all import *
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from winak.globaloptimization.betterhopping import BetterHopping

adsorbate = '../testsystems/rea.xyz'
molecule = read(adsorbate)

#NOTE: Define your own calculator here!
molecule.set_calculator(EMT())

bh = BetterHopping(atoms=molecule,
                  temperature=100 * kB,
                  dr=0.55,
                  optimizer=BFGS,
                  fmax=4,
                  logfile='tt.log',
                  maxmoves=50,
                  movemode=1,
                  numdelocmodes=3
                  )
bh.run(50)


