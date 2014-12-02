import numpy as np
from ase.all import *

from winak.curvilinear.InternalCoordinates import icSystem
from winak.curvilinear import Coordinates
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC

from winak.globaloptimization.delocalizer import *

m =read('testsystems/clethen-on-Ag.traj')

e=Delocalizer(m)

cell = m.get_cell()

coords=CDC(e.x_ref.flatten(), e.masses, atoms=e.atoms, \
             ic=e.ic, u=e.get_U(), unit=1.0, cell=cell) #only use the cell=cell command if you want "vibrations" of the cell as well
coords.write_jmol('dol') #delocalizing out loud
