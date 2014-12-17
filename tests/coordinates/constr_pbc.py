from ase.all import *
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.surface import fcc100, fcc111
import numpy as np

system = fcc100('Pd', (2,2,3), a=3.94, vacuum=10.)

natoms =len(system) 
np.set_printoptions(threshold=np.nan)
import time
start=time.time()
print start

#constraints need to be satisfied on two levels
#first we need to partition our active coordinate space 
#into constrained and unconstrained coordinates, this part 
#happens in the Delocalizer
#secondly the backiteration needs to remove components of 
#constrained coordinates. For cartesian or lattice coordinates this 
#amounts to zeroing columns of the Bmatrix (passed to the backIteration 
#        via biArgs col_constraints=[column1, column2]
#For internal coordinates we zero the rows of the Bmatrix, passed by 
#row_constraints=[row1,row2]

#pic example
run_1 = False 
run_2 = True 
run_3 = False

#Example 1:
#    here we constrain the unit cell

if run_1:

    d = Delocalizer(system, periodic=True, dense=False, weighted=False, \
                        add_cartesians = False)

    e = d.constrainCell()
    d.constrain(e)

    nx = d.ic.nx

    #the last six coordinates in u correspond to the pure cell changes
    d.ic.backIteration = d.ic.denseBackIteration
    coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_constrained_U(),
            biArgs={'RIIS': False, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6, 
                'col_constraints': [nx,nx+1,nx+2,nx+3,nx+4,nx+5,nx+6,nx+7,nx+8] ,
                #needed for sparseBackIteration...empty columns or rows make cholesky very messy
                #'iclambda': 20.5,
                })

    coords.write_jmol('cell_constrained_Pd2x2x4.jmol')

    c0 = coords.cell.copy()
    print 'c0 ',c0
    X0 = system.positions.flatten()
    coords.s[:-6]= 100.0
    X1 = coords.getX()
    print 'c ',coords.cell
    print '-------'
    print 'c-c0 ', coords.cell-c0 

#Example 2:
#    here we constrain the cartesian positions of the lowest layer
#    of atoms 0,1,2,3

if run_2:
    
    d = Delocalizer(system, periodic=True, dense=False, weighted=False, \
                        add_cartesians = True)

    e = d.constrainAtoms([
        [0,0],[0,1],[0,2],
        [1,0],[1,1],[1,2],
        [2,0],[2,1],[2,2],
        [3,0],[3,1],[3,2],
        ])
    #d.constrain(e)

    constraints = [0,1,2,3,4,5,6,7,8,9,10,11] 

    #the last six coordinates in u correspond to the pure cell changes
    d.ic.backIteration = d.ic.denseBackIteration
    #coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_constrained_U(),
    coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_U(),
            biArgs={'RIIS': False, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6, 
                #'col_constraints': constraints,
                #needed for sparseBackIteration...empty columns or rows make cholesky very messy
                #'iclambda': 20.5,
                })
    
    coords.write_jmol('cart_constrained_Pd2x2x4.jmol')

    c0 = coords.x[:12]
    print 'x0 ',c0
    coords.s[:]= 100.0
    coords.getX
    print 'x ',coords.x[:12]
    print '-------'
    print 'x-x0 ', c0 - coords.x[:12] 
