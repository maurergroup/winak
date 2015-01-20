from ase.all import *
from ase.calculators.dftb import Dftb
from ase.optimize import BFGS
from winak.globaloptimization.betterhopping import Delocalizer 
from winak.globaloptimization.betterhopping import BetterHopping
from ase.constraints import FixInternals

adsorbate = '../testsystems/rea.xyz'
d='/home/reini/Work/codes/QM-codes/hotbit/param/SURF/DFTB+_2/'
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

deloc = Delocalizer(molecule)
#constraining all stretches with FixInternals
stretch_ind = deloc.ic.getStretchBendTorsOop()[0][0]
stretches = deloc.ic.getStretchBendTorsOop()[1]
bonds = []
for i in range(len(stretch_ind)):
    i, j = stretches[i] -1
    print i, j
    bonds.append([molecule.get_distance(i,j), [i,j]])

f = FixInternals(molecule, bonds=bonds, angles=[], dihedrals=[])
molecule.set_constraint(f)

bh = BetterHopping(atoms=molecule,
                  temperature=100 * kB,
                  dr=1.0,
                  optimizer=BFGS,
                  fmax=0.025,
                  logfile='tt.log',
                  maxmoves=50,
                  movemode=1,
                  numdelocmodes=20,
                  constrain=True
                  )
bh.run(50)


