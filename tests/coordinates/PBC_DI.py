from ase.all import *
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import Set_of_CDCs
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.surface import fcc100
import numpy as np

#system = fcc(directions=[[1,0,0], [0,1,0], [0,0,1]],
            #size=(1,1,1), symbol='Pd' )

system = fcc100('Pd', (2,2,4), a=3.94, vacuum=10.)

#masses = system.get_masses()
#atoms = system.get_chemical_symbols()
#x_ref = system.positions
#x0 = x_ref.flatten()
#cell = system.get_cell()

#pvcg = PVCG(atoms=atoms, masses=masses, cell = cell)
#iclist = pvcg(x0)
#ic2 = Periodic_icSystem(iclist, len(atoms), masses=masses, xyz=x0, cell=cell)
#ic2.backIteration = ic2.denseBackIteration
#print ic2.Bnnz, ic2.n, ic2.nx+9
#natoms =len(system) 

#from winak.curvilinear.numeric.SparseMatrix import AmuB, svdB, eigB
#B=ic2.B
#Bt=ic2.evalBt(perm=0)
#G=AmuB(B,Bt)
#v2, ww, u=svdB(G, k=3*natoms)
##ww, u=eigB(G, k=3*natoms)
##u = np.real(u.transpose()[:3*natoms])
#print 'E ',(ww[:3*natoms])

#coords = PC(x0,masses,unit=1.0,atoms=atoms,ic=ic2, Li=u)

d1 = Delocalizer(system, periodic=False) 
d = Delocalizer(system, periodic=True) 
coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_U())


coords.write_jmol('s2.jmol')

#coords.s=np.random.random(3*natoms)
#coords.s2x()

#system.positions = coords.x.reshape(-1,3)
#system.cell = coords.cell.reshape(-1,3)
#view(system)
#print coords.x 
#print coords.cell

#coords.x2s()
#coords.s2x()
