from winak.screening.criterion import Metropolis
from winak.screening.displacer import MultiDI
from winak.screening.energyevaluator import potEE
from winak.screening.ultimatescreener import UltimateScreener
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.io import read
from ase.units import kB

"""NOTE: THIS WILL OFTEN FAIL WITH THE EMT CALCULATOR"""

rea=read('../testsystems/dis.xyz')
rea.set_calculator(EMT())

rea.get_potential_energy()

crit=Metropolis(T=100*kB)

disp=MultiDI(stepwidth=0.7,
        constrain=True, 
        numdelocmodes=0.25)#,loghax=True)
ee=potEE(EMT(),BFGS,fmax=1.0)

us=UltimateScreener(rea,ee,disp,crit)
us.run(5)
