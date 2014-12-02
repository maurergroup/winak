import numpy as np
from ase.all import *

from winak.curvilinear.InternalCoordinates import icSystem
from winak.curvilinear import Coordinates
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC

from winak.globaloptimization.delocalizer import *

m =read('clethen.xyz')
e=Delocalizer(m)

coords=DC(e.x_ref.flatten(), e.masses, internal=True, atoms=e.atoms, \
             ic=e.ic, L=None, Li=None,u=e.get_U())
coords.write_jmol('dol') #delocalizing out loud
