from ase.all import *
from ase.calculators.dftb import Dftb
from ase.optimize import BFGS
from winak.globaloptimization.betterhopping import BetterHopping

adsorbate = '../testsystems/rea.xyz'
d='/home/konstantin/software/DFTB/param/SURF/'
molecule = read(adsorbate)

calc_dftb = Dftb(label='dftb',atoms=molecule,
                 Hamiltonian_MaxAngularMomentum_ = '',
                 Hamiltonian_MaxAngularMomentum_C = '"p"',
                 Hamiltonian_MaxAngularMomentum_H = '"s"',
                 Hamiltonian_MaxAngularMomentum_O = '"p"',
                 Hamiltonian_MaxSCCIterations = 2000,
                 Hamiltonian_SCC ='YES',
                 Hamiltonian_SCCTolerance = 1.0E-007,
                 Hamiltonian_SlaterKosterFiles_Prefix=d,
                 )

molecule.set_calculator(calc_dftb)

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


