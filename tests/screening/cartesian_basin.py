from winak.screening.criterion import Metropolis
from winak.screening.displacer import Cartesian
from winak.screening.energyevaluator import potEE
from winak.screening.ultimatescreener import UltimateScreener
from ase.optimize import BFGS
from ase.all import *
from ase.units import kB

rea=read('../testsystems/rea.xyz')
rea.set_calculator(EMT())

crit=Metropolis(T=1000*kB)
disp=Cartesian(stepwidth=0.7)
ee=potEE(EMT(),BFGS,fmax=1.0)

us=UltimateScreener(rea,ee,disp,crit)
us.run(10)
