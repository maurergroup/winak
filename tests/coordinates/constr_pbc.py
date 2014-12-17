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

#system = fcc(directions=[[1,0,0], [0,1,0], [0,0,1]],
            #size=(1,1,1), symbol='Pd' )
system = fcc100('Pd', (2,2,1), a=3.94, vacuum=10.)

#system = read('../testsystems/rea.xyz')

#system.cell[1,1] = 10

natoms =len(system) 
np.set_printoptions(threshold=np.nan)
import time
start=time.time()
print start
print 'periodic'

d = Delocalizer(system, periodic=True, dense=True, weighted=False, \
        add_cartesians = False)
e1 = []
#e1 = d.constrainAtoms([0,1])
#e1 = d.constrainCell()
#e2 = d.constrainAtoms([
            #[0,0],[0,1],[0,2],
            #[1,0],[1,1],[1,2],
            #[2,0],[2,1],[2,2],
            #[3,0],[3,1],[3,2],
            #[4,0],[4,1],[4,2],
            #[5,0],[5,1],[5,2],
            #[6,0],[6,1],[6,2],
            #[7,0],[7,1],[7,2],
            #[8,0],[8,1],[8,2],
            #])
#e1 = d.constrainStretches2([0,1,2,3])
#e = e1 #+ e2
#print e
#d.constrain2(e)
#d.u2 holds the constrained DI vecs.
#print d.ww
print time.time() - start

#coords = DC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, u=d.get_constrained_U())
#coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_constrained_U(),
coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_U(),
        #biArgs={'iclambda':1e-8, 'RIIS': True, 'maxiter': 100, 'eps': 1e-6, 'maxEps':1e-6})
        biArgs={'RIIS': True, 'maxiter':10})

print len(d.ww)
print d.ww
u2 = d.u
for i in range(len(u2)):
    print np.dot(u2[i],u2[0])

coords.write_jmol('s2.jmol')

print 'c0 ',coords.cell
X0 = system.positions.flatten()
coords.s[:]= 1.0
X1 = coords.getX()
print 'c ',coords.cell
print 'x ',X1-X0
print 's ',coords.getS(X1)
