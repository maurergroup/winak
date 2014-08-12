#Coordinate system for gasphase Azobenzene
import numpy as np
from ase.all import *
import os
from os.path import join as pathjoin

#reference geometry
#x_ref = np.array([
       #[  2.50009222e+00,  -4.47418403e-01,  -1.47579370e-02],
       #[  4.01003603e+00,  -2.43935173e+00,  -1.46655159e-02],
       #[  3.04556131e+00,  -4.73183577e+00,   1.67121835e-03],
       #[  5.74729068e-01,  -5.04380122e+00,   1.76991800e-02],
       #[ -9.25734242e-01,  -3.03466758e+00,   1.75511233e-02],
       #[  9.25762351e-01,   3.03475050e+00,  -1.40260061e-02],
       #[ -5.74738077e-01,   5.04385734e+00,  -1.04569778e-02],
       #[ -3.04557396e+00,   4.73182681e+00,   1.89602652e-03],
       #[ -4.01003914e+00,   2.43928126e+00,   1.01061327e-02],
       #[ -2.50006971e+00,   4.47387462e-01,   6.86467132e-03],
       #[  2.09639692e+00,  -1.45845162e+00,  -7.81569390e-03],
       #[  2.92750622e+00,  -2.57194998e+00,  -7.66420475e-03],
       #[  2.38495524e+00,  -3.86451380e+00,   1.51885951e-03],
       #[  9.99763705e-01,  -4.04005654e+00,   1.05340871e-02],
       #[  1.59151626e-01,  -2.92818149e+00,   1.04680748e-02],
       #[ -1.59128661e-01,   2.92823157e+00,  -8.70520950e-03],
       #[ -9.99763736e-01,   4.04008831e+00,  -6.71877333e-03],
       #[ -2.38495775e+00,   3.86451030e+00,   1.89938146e-04],
       #[ -2.92750936e+00,   2.57192046e+00,   4.90442492e-03],
       #[ -2.09637721e+00,   1.45844077e+00,   3.13847972e-03],
       #[  6.99718862e-01,  -1.63311570e+00,   1.36386214e-03],
       #[ -6.99692748e-01,   1.63313375e+00,  -3.44684500e-03],
       #[ -2.46764705e-01,  -5.79634448e-01,   2.78769556e-03],
       #[  2.46792038e-01,   5.79651149e-01,  -4.87032884e-03]
#])
#x0 = x_ref.flatten()

#atoms_object = Atoms('H10C12N2', positions = x_ref)
#masses  = atoms_object.get_masses()
#atoms = atoms_object.get_chemical_symbols()

##internal coordinate definition

##the first coordinates are active
##all others will be fixed, also during
##the optimization
#internals = np.array([
    #1, 23, 24,  #N1 - N2      #ACTIVE
    #2, 22, 23, 24, #alpha1    #ACTIVE
    #2, 21, 24, 23, #alpha2    #ACTIVE
    #3, 21, 23, 24, 22, #omega #ACTIVE
    #1, 21, 23,  #N1 - C11     #NOT ACTIVE, but mapped
    #1, 22, 24,  #N2 - C12     #NOT ACTIVE
    #3, 11, 21, 23, 24, #beta1 #NOT ACTIVE
    #3, 16, 22, 24, 23, #beta2 #NOT ACTIVE
    #1, 11, 21,
    #1, 12, 11,
    #1, 13, 12,
    #1, 14, 13,
    #1, 15, 14,
    #1, 1,  11,
    #1, 2,  12,
    #1, 3,  13,
    #1, 4,  14,
    #1, 5,  15,
    #1, 16, 22,
    #1, 17, 16,
    #1, 18, 17,
    #1, 19, 18,
    #1, 20, 19,
    #1, 6,  16,
    #1, 7,  17,
    #1, 8,  18,
    #1, 9,  19,
    #1, 10, 20,
    #2, 11, 23, 21,
    #2, 16, 24, 22,
    #2, 12, 21, 11,
    #2, 13, 11, 12,
    #2, 14, 12, 13,
    #2, 15, 13, 14,
    #2, 17, 22, 16,
    #2, 18, 16, 17,
    #2, 19, 17, 18,
    #2, 20, 18, 19,
    #2,  1, 21, 11,
    #2,  2, 11, 12,
    #2,  3, 12, 13,
    #2,  4, 13, 14,
    #2,  5, 14, 15,
    #2,  6, 22, 16,
    #2,  7, 16, 17,
    #2,  8, 17, 18,
    #2,  9, 18, 19,
    #2, 10, 19, 20,
    #3, 12, 11, 21, 23,
    #3, 13, 12, 11, 21,
    #3, 14, 13, 12, 11,
    #3, 15, 14, 13, 12,
    #3, 17, 16, 22, 24,
    #3, 18, 17, 16, 22,
    #3, 19, 18, 17, 16,
    #3, 20, 19, 18, 17,
    #3,  1, 11, 21, 15,
    #3,  2, 12, 11,  1,
    #3,  3, 13, 12,  2,
    #3,  4, 14, 13,  3,
    #3,  5, 15, 14,  4,
    #3,  6, 16, 22, 20,
    #3,  7, 17, 16,  6,
    #3,  8, 18, 17,  7,
    #3,  9, 19, 18,  8,
    #3, 10, 20, 19,  9,
    #0
#])

from INTERNALS.curvilinear.InternalCoordinates import icSystem
from INTERNALS.curvilinear import Coordinates
from INTERNALS.curvilinear.Coordinates import InternalEckartFrameCoordinates as IEFC

from INTERNALS.globaloptimization.delocalizer import *

m =read('clethen.xyz')
e=Delocalizer(m)
#Delocali

Li=e.u
#this Li is crap
L = np.dot(np.linalg.inv(np.dot(Li,Li.transpose())),Li)
L = L.transpose()
#L=np.identity(len(masses)*3-6)

#ic = icSystem(internals, len(atoms), masses = masses, xyz = x0.copy())
#ic.backIteration = ic.denseBackIteration

coords= IEFC(e.x_ref.flatten(), e.masses, internal=True, atoms=e.atoms, \
             ic=e.ic, L=L, Li=Li)


# ss = coords.s
# ss[2] = 20
# m.positions=(coords.getX(ss)).reshape(-1,3)
# view(m)


