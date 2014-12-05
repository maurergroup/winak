from ase.all import *
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.Coordinates import InternalCoordinates as IC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import Set_of_CDCs

from ase.lattice.surface import fcc100

system = fcc100('Pd', (2,2,2), a=3.94, vacuum=10.)

from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

masses = system.get_masses()
atoms = system.get_chemical_symbols()
x_ref = system.positions
x0 = x_ref.flatten()
cell = system.get_cell()

print 'periodic'
pvcg = PVCG(atoms=atoms, masses=masses, cell = cell)
iclist = pvcg(x0)
np.set_printoptions(threshold='nan')
print iclist

#mbig = pvcg.masses
#xbig = pvcg.xyz_pbc
#ic = icSystem(iclist, 8*len(atoms), masses=mbig, xyz=xbig)
#ic.backIteration = ic.denseBackIteration
#print ic.Bnnz, ic.n, ic.nx

ic2 = Periodic_icSystem(iclist, len(atoms), masses=masses, xyz=x0, cell=cell)
ic2.backIteration = ic2.denseBackIteration
print ic2.Bnnz, ic2.n, ic2.nx+9

#Periodic ic INIT works!
#import numpy as np
natoms =1 
#m=np.identity(natoms*3+9)
#b=ic2.B.full()
#g=np.dot(b,m)
#g=np.dot(g,b.transpose())

#v2,ww,u=np.linalg.svd(g)
#u = u[:3*natoms]
#print ww[:3*natoms]

#coords = DC(x0,masses,unit=1.0,atoms=atoms,ic=ic2, u=u)
#coords.write_jmol('s2.jmol')

print(ic2)

#coords.x2s()
#coords.s2x()
