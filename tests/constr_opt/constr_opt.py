from ase.io import read
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from winak.globaloptimization.betterhopping import Delocalizer 
from winak.globaloptimization.betterhopping import BetterHopping
from ase.constraints import FixInternals

adsorbate = '../testsystems/rea.xyz'
molecule = read(adsorbate)

#NOTE: you will need to set a sensible calculator for this to give results!
molecule.set_calculator(EMT())

deloc = Delocalizer(molecule)

assert 0

#constraining all stretches with FixInternals
stretch_ind = deloc.ic.getStretchBendTorsOop()[0][0]
stretches = deloc.ic.getStretchBendTorsOop()[1]
bonds = []
for i in range(len(stretch_ind)):
    i, j = stretches[i] -1
    print i, j
    bonds.append([molecule.get_distance(i,j), [i,j]])

f = FixInternals(bonds=bonds, angles=[], dihedrals=[])
#molecule.set_constraint(f)

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


