from winak.screening.criterion import Metropolis
from winak.screening.displacer import Celltrans
from winak.screening.energyevaluator import potEE
from winak.screening.ultimatescreener import UltimateScreener
from ase.optimize import BFGS
from ase.all import *
from ase.units import kB

"""NOTE: THIS WILL OFTEN FAIL WITH THE EMT CALCULATOR"""

rea=read('../testsystems/C6H6_on_Ag.traj')
rea.set_calculator(EMT())
f = FixAtoms(mask=[atom.symbol == 'Ag' for atom in rea])
rea.set_constraint(f)

crit=Metropolis(T=100*kB)
disp=Celltrans(stepwidth=1.2,adsorbate=(18,30),cell_scale=[0.5,0.3,0.04])
ee=potEE(EMT(),BFGS,fmax=1.0)

us=UltimateScreener(rea,ee,disp,crit)
us.run(10)
