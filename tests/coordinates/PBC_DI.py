from ase.all import *
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import Set_of_CDCs

from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.surface import fcc100

system = fcc(directions=[[1,0,0], [0,1,0], [0,0,1]],
            size=(1,1,1), symbol='Pd' )

system = fcc100('Pd', (2,2,2), a=3.94, vacuum=10.)
#system = fcc100('Pd', (1,1,1), a=3.94, vacuum=10.)

from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

masses = system.get_masses()
atoms = system.get_chemical_symbols()
x_ref = system.positions
x0 = x_ref.flatten()
cell = system.get_cell()

pvcg = PVCG(atoms=atoms, masses=masses, cell = cell)
iclist = pvcg(x0)
np.set_printoptions(threshold='nan')

print iclist
print pvcg.oops

print 'isolated'
mbig = pvcg.masses
xbig = pvcg.xyz_pbc
ic = icSystem(iclist, 8*len(atoms), masses=mbig, xyz=xbig)
ic.backIteration = ic.denseBackIteration
print ic.Bnnz, ic.n, ic.nx

print 'periodic'
ic2 = Periodic_icSystem(iclist, len(atoms), masses=masses, xyz=x0, cell=cell)
ic2.backIteration = ic2.denseBackIteration
print ic2.Bnnz, ic2.n, ic2.nx+9

#Periodic ic INIT works!

natoms =len(system) 

import numpy as np
#from winak.curvilinear.numeric.SparseMatrix import AmuB,svdB
#B=ic2.B
#Bt=ic2.Bt
#G=AmuB(B,Bt)

#v2, ww, u=svdB(G, k=3*natoms+9)
#u = u[:3*natoms+9]
#print (ww[:3*natoms+9])[::-1]

b = ic2.B.full()
g = np.dot(b, b.transpose())
v2,ww,u=np.linalg.svd(g)
u = u[:3*natoms]
print ww[:3*natoms]

coords = PC(x0,masses,unit=1.0,atoms=atoms,ic=ic2, Li=u)

view(system)
print coords.x
print coords.cell

coords.write_jmol('s2.jmol')

#coords.s=np.random.random(3*natoms)
#coords.s2x()


#print(ic2)

#system.positions = coords.x.reshape(-1,3)
#system.cell = coords.cell.reshape(-1,3)
#view(system)
#print coords.x 
#print coords.cell

#coords.x2s()
#coords.s2x()
