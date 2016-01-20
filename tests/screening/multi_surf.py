from winak.screening.criterion import Metropolis
from winak.screening.displacer import MultiDI
from winak.screening.energyevaluator import potEE
from winak.screening.ultimatescreener import UltimateScreener
from ase.optimize import BFGS
from ase.all import *
from ase.units import kB

"""NOTE: THIS WILL OFTEN FAIL WITH THE EMT CALCULATOR"""

rea=read('../testsystems/C6H6_on_Ag_dis.traj')
rea.set_calculator(EMT())
f = FixAtoms(mask=[atom.symbol == 'Ag' for atom in rea])
rea.set_constraint(f)

crit=Metropolis(T=100*kB)
disp=MultiDI(stepwidth=1.7,numdelocmodes=0.33,constrain=False,adsorbate=(18,30),cell_scale=[0.5,0.3,0.031],loghax=True)
ee=potEE(EMT(),BFGS,fmax=1.0)

us=UltimateScreener(rea,ee,disp,crit)
us.run(5)
