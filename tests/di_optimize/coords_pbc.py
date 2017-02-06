from ase.io import read
from ase.calculators.emt import EMT
from ase.visualize import view
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.surface import fcc100, fcc111
import numpy as np

system = read('ammonia.gen')
system.set_calculator(EMT())
natoms =len(system) 
np.set_printoptions(threshold=np.nan)
import time
start=time.time()
print start

d = Delocalizer(system, periodic=True, dense=True, weighted=True, \
                    add_cartesians = False)

print 'timing 1 ',time.time() - start

coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_U(),
        # biArgs={'iclambda':1e-8, 'RIIS': True, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6})
        biArgs={'RIIS': True, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6})

print len(d.ww)
print d.ww

print 'timing 2 ', time.time() - start

# coords.write_jmol('s2.jmol')
# print 'timing 3', time.time() - start

# view(system)

print 'c0 ',coords.cell
X0 = system.positions.flatten()
coords.s[:]= 0.0
coords.s[-2]= 10.0
tmp = coords.getX()
X1 = tmp[:-9]
cell = tmp[-9:]
# print 'c ',coords.cell
print 'c1', cell
print 'x ',X1-X0
print 's ',len(coords.s), coords.getS(tmp)

system.cell = cell.reshape([3,3])
system.positions = X1.reshape([-1,3])

f = np.zeros(len(X1))#system.get_forces()
s = np.zeros(9)
print f
print s

print '-----'
fs = coords.grad_x2s(np.concatenate([f.flatten(),s.flatten()]),gradientProps={'iclambda':0.00001})
print 'fs', len(fs)
print fs


# view(system)
