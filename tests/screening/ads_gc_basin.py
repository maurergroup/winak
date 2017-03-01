from winak.screening.criterion import Metropolis
from winak.screening.displacer import GC
from winak.screening.energyevaluator import grandEE
from winak.screening.ultimatescreener import UltimateScreener
from ase.optimize import BFGS
from ase.all import *
from ase.units import kB

"""NOTE: THIS WILL OFTEN FAIL WITH THE EMT CALCULATOR"""

rea=read('../../testsystems/C6H6_on_Ag.traj')
rea.set_calculator(EMT())
#f = FixAtoms(mask=[atom.symbol == 'Ag' for atom in rea])
### better pass constraints as index list
f = FixAtoms(indices=[atom.index for atom in rea if atom.symbol == 'Ag'])
rea.set_constraint(f)

for atom in rea:
    if 17 < atom.index < 31:
        atom.tag=100

### reference energies
eag=0
eh=0 
ec=0 
eclean=0

mu={'Ag':0.0, 'C':0.0, 'H':0.0}
ecomp={'Ag':eag, 'C':ec, 'H':eh}

crit=Metropolis(T=100*kB)
disp=GC(prob=1,stepwidth=0.7,numdelocmodes=8,constrain=True,adsorbate=100,cell_scale=[0.5,0.3,0.1],atm='C')
ee=grandEE(EMT(),BFGS,fmax=1.0, ecomp=ecomp, mu=mu, eref=eclean, adsorbate=100)

us=UltimateScreener(rea,ee,disp,crit)
us.run(30)
