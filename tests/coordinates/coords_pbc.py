from ase.all import *
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.surface import fcc100, fcc111
import numpy as np

system = fcc100('Pd', (2,2,2), a=3.94, vacuum=10.)

natoms =len(system) 
np.set_printoptions(threshold=np.nan)
import time
start=time.time()
print start

d = Delocalizer(system, periodic=True, dense=False, weighted=False, \
                add_cartesians = True)

print 'timing 1 ',time.time() - start

coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_U(),
        biArgs={'iclambda':1e-8, 'RIIS': True, 'maxiter': 100, 'eps': 1e-6, 'maxEps':1e-6})

print len(d.ww)
print d.ww

print 'timing 2 ', time.time() - start

coords.write_jmol('s2.jmol')

print 'timing 3', time.time() - start

print 'c0 ',coords.cell
X0 = system.positions.flatten()
coords.s[:]= 100.0
X1 = coords.getX()
print 'c ',coords.cell
print 'x ',X1-X0
print 's ',coords.getS(X1)
