import numpy as np
from ase.all import *


from INTERNALS.curvilinear.InternalCoordinates import icSystem
from INTERNALS.curvilinear import Coordinates
from INTERNALS.curvilinear.Coordinates import DelocalizedCoordinates as DC

from INTERNALS.globaloptimization.delocalizer import *

m =read('clethen.xyz')
e=Delocalizer(m)

coords=DC(e.x_ref.flatten(), e.masses, internal=True, atoms=e.atoms, \
             ic=e.ic, L=None, Li=None,u=e.u)
coords.write_jmol('dol') #delocalizing out loud
