from ase.all import *
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.surface import fcc100, fcc111
import cPickle as pickle
import numpy as np

i = 2
j = 2
k = 3

system = fcc100('Pd', (i,j,k), a=3.94, vacuum=10.)

natoms =len(system) 
np.set_printoptions(threshold=np.nan)
import time
start=time.time()
print start

iclist = None
iclist = pickle.load(open("iclist.p", "rb"))

d = Delocalizer(system, icList=iclist, periodic=True, dense=False, weighted=False, \
                    add_cartesians = True)
e1 = []
e1 = d.constrainCell()
cart = []
for c in range(i*j):
    cart.append([c,0])
    cart.append([c,1])
    cart.append([c,2])
e2 = d.constrainAtoms(cart)
e2 = []
e = e1 + e2
d.constrain(e)

print 'ww ', d.ww

pickle.dump(d.iclist, open("iclist.p", "wb"))

print time.time() - start

print 'there are that many constraints ', len(e)
print 'there are that many DIs ', len(d.u)
print 'that leaves that many independent DIs ', len(d.u2)-len(e)
d.u2 = d.u2[len(e):]

coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_constrained_U(),
        biArgs={'RIIS': True, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6, 
            'col_constraints': d.constraints_cart,
            'iclambda': 1e-6,
            })

print time.time() - start
print 'Calculating vectors for jmol visualization'
coords.write_jmol('pes.jmol')

print time.time() - start

print 'X0  ', coords.x
print 'c0 ', coords.cell
coords.s[len(e):]= 1.0
coords.getX()
print 'X1  ', coords.x
print 'c1 ', coords.cell

print time.time() - start
