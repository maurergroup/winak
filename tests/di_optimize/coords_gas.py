import numpy as np
from ase.io import read
from ase.visualize import view
from obcalc import OBForceField
from ase.optimize import BFGS

from winak.curvilinear.InternalCoordinates import icSystem
from winak.curvilinear import Coordinates
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC

from winak.globaloptimization.delocalizer import *

m =read('../testsystems/clethen.xyz')

calc = OBForceField()
m.set_calculator(calc)
dyn = BFGS(m)
dyn.run(0.001)

print 'Cartesian optimization finished...'

#TEST ICLIST
iclist = [1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 2, 5, 1, 2, 6, 2, 1, 5, 2, 
          2, 2, 4, 1, 2, 3, 4, 1, 2, 5, 6, 2, 2, 2, 3, 1, 2, 1,
          6, 2, 3, 3, 1, 2, 5, 3, 3, 1, 2, 6, 3, 4, 1, 2, 6, 3,
          4, 1, 2, 5, 4, 2, 3, 4, 1, 4, 6, 1, 5, 2, 0]

#DENSE
d=Delocalizer(m, icList=iclist, periodic=False, dense=True, weighted=True)#, weighted=True)

coords=DC(d.x_ref.flatten(), d.masses, internal=True, atoms=d.atoms, \
        ic=d.ic, L=None, Li=None,u=d.get_U(), #)#, biArgs={'iclambda' :0.0001})
        biArgs={'RIIS': True, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-7,
            # 'iclambda' : 1e-6, 
            'RIIS_dim': 4, 'RIIS_maxLength':6, 'maxStep':0.5,})
coords.write_jmol('dol') #delocalizing out loud

print 'constructed DI coordinates'

# view(m)

    

mlist = [m.copy()]

for ss in np.linspace(0,100,10):
    #transform from DCs to CC displacements
    coords.s[:] = 0.0
    coords.s[-2] =  ss
    coords.s2x()
    X1 = coords.x
    print 's'
    print coords.getS(X1)
    # print 'X'
    # print X1

    m.positions = X1.reshape([-1,3])
    
    mlist.append(m.copy())
    
    f = m.get_forces()
     # print 'fx'
     # print f
    fs = coords.grad_x2s(f.flatten(),gradientProps={'iclambda':0.10010 })
    print 'fs'
    print fs

view(mlist)


#transform from CCs to DC displacements
#X1[0.2] += 0.1
#print coords.getS(X1) 

#


assert 0


#SPARSE
d=Delocalizer(m, icList=iclist, periodic=False, dense=False, weighted=True)#, weighted=True)

coords=DC(d.x_ref.flatten(), d.masses, internal=True, atoms=d.atoms, \
        ic=d.ic, L=None, Li=None,u=d.get_U(),unit=1.0,
        biArgs={'RIIS': True, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6})
coords.write_jmol('dol2') #delocalizing out loud

coords.s[:] =  2.000
coords.s2x()
X2 = coords.x
print coords.getS(X2)
print X2
print '-----'
print X1-X2
