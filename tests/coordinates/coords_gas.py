import numpy as np
from ase.io import read
from winak.curvilinear.InternalCoordinates import icSystem
from winak.curvilinear import Coordinates
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC

from winak.globaloptimization.delocalizer import *

m =read('../testsystems/clethen.xyz')

#TEST ICLIST
iclist = [1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 2, 5, 1, 2, 6, 2, 1, 5, 2, 
          2, 2, 4, 1, 2, 3, 4, 1, 2, 5, 6, 2, 2, 2, 3, 1, 2, 1,
          6, 2, 3, 3, 1, 2, 5, 3, 3, 1, 2, 6, 3, 4, 1, 2, 6, 3,
          4, 1, 2, 5, 4, 2, 3, 4, 1, 4, 6, 1, 5, 2, 0]

#DENSE
d=Delocalizer(m, icList=iclist, periodic=False, dense=True, weighted=True)#, weighted=True)

coords=DC(d.x_ref.flatten(), d.masses, internal=True, atoms=d.atoms, \
        ic=d.ic, L=None, Li=None,u=d.get_U())#, biArgs={'iclambda' :0.0001})
coords.write_jmol('dol') #delocalizing out loud

coords.s[:] =  2.000
coords.s2x()
X1 = coords.x
print coords.getS(X1)
print X1

#SPARSE
d=Delocalizer(m, icList=iclist, periodic=False, dense=False, weighted=False)#, weighted=True)

coords=DC(d.x_ref.flatten(), d.masses, internal=True, atoms=d.atoms, \
        ic=d.ic, L=None, Li=None,u=d.get_U(), biArgs={'iclambda':0.000001})
coords.write_jmol('dol2') #delocalizing out loud

coords.s[:] =  2.000
coords.s2x()
X2 = coords.x
print coords.getS(X2)
print X2
print '-----'
print X1-X2
