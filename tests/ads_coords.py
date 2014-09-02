import numpy as np
from ase.all import *


from INTERNALS.curvilinear.InternalCoordinates import icSystem
from INTERNALS.curvilinear import Coordinates
from INTERNALS.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC

from INTERNALS.globaloptimization.delocalizer import *

m =read('clethen-on-Ag.traj')

e=Delocalizer(m)

cell = m.get_cell()

coords=CDC(e.x_ref.flatten(), e.masses, atoms=e.atoms, \
             ic=e.ic, u=e.get_U(), unit=1.0 cell=cell)
coords.write_jmol('dol') #delocalizing out loud
