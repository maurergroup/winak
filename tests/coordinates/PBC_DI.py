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
system = fcc100('Pd', (2,2,2), a=3.94, vacuum=10.)


#system.cell[1,1] = 10

natoms =len(system) 

import time
start=time.time()
print start
print 'periodic'

d = Delocalizer(system, periodic=True, dense=False, weighted=True)
print time.time() - start

coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_U(),
        biArgs={'iclambda':1e-6, 'RIIS': False, 'maxiter': 500})

end = time.time()
print end-start

coords.write_jmol('s2.jmol')

view(system)
coords.s[:]= 5.0
X1 = coords.getX()
system.positions = coords.x.reshape(-1,3)
system.cell = coords.cell.reshape(-1,3)
view(system)
