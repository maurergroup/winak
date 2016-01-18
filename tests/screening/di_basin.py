from winak.screening.criterion import Metropolis
from winak.screening.displacer import DI
from winak.screening.energyevaluator import potEE
from winak.screening.ultimatescreener import UltimateScreener
from ase.optimize import BFGS
from ase.all import *
from ase.units import kB

rea=read('../testsystems/rea.xyz')
rea.set_calculator(EMT())

crit=Metropolis(T=100*kB)
disp=DI(stepwidth=0.7,numdelocmodes=0.25,constrain=True)
ee=potEE(EMT(),BFGS,fmax=1.0)

us=UltimateScreener(rea,ee,disp,crit)
us.run(10)
