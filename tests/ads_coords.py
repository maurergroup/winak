import numpy as np
from ase.all import *


from INTERNALS.curvilinear.InternalCoordinates import icSystem
from INTERNALS.curvilinear import Coordinates
from INTERNALS.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC

from INTERNALS.globaloptimization.delocalizer import *

m =read('clethen.xyz')
e=Delocalizer(m)

coords=CDC(e.x_ref.flatten(), e.masses, atoms=e.atoms, \
             ic=e.ic, u=e.get_U())
coords.write_jmol('dol') #delocalizing out loud
