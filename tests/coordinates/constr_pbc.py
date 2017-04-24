from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.InternalCoordinates import icSystem, Periodic_icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG 
from winak.curvilinear.InternalCoordinates import PeriodicValenceCoordinateGenerator as PVCG 

from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.surface import fcc100, fcc111
import numpy as np

system = fcc100('Pd', (2,2,4), a=3.94, vacuum=10.)

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
                        add_cartesians = False)

    e2 = d.constrainStretches([0,1])
    e = e2
    d.constrain(e)
    print 'there are that many stretches ', len(e)
    print 'there are that many DIs ', len(d.u)
    print 'that leaves that many independent DIs ', len(d.u2)-len(e)
    #the last six coordinates in u correspond to the pure cell changes
    d.ic.backIteration = d.ic.denseBackIteration
    coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_constrained_U(),
            biArgs={'RIIS': True, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6,'maxStep' :0.25,
                'RIIS_dim':14,'RIIS_maxLength':10,
                'col_constraints': d.constraints_cart, 'row_constraints':d.constraints_int,
                #needed for sparseBackIteration...empty columns or rows make cholesky very messy
                #'iclambda': 1e-6,
                })
    
    coords.write_jmol('stretch_constrained_Pd2x2x4.jmol')

    dd = np.linalg.norm(coords.x.reshape(-1,3)[0] - coords.x.reshape(-1,3)[1])
    print 'd  ', dd
    coords.s[:]= 10.0
    coords.getX()
    dd = np.linalg.norm(coords.x.reshape(-1,3)[0] - coords.x.reshape(-1,3)[1])
    print 'd  ', dd
    
if run_3:
    
    d = Delocalizer(system, periodic=True, dense=False, weighted=False, \
                        add_cartesians = True)

    e1 = d.constrainCell()
    e2 = d.constrainAtoms([
        [0,0],[0,1],[0,2],
        [1,0],[1,1],[1,2],
        [2,0],[2,1],[2,2],
        [3,0],[3,1],[3,2],
        [4,0],[4,1],[4,2],
        [5,0],[5,1],[5,2],
        [6,0],[6,1],[6,2],
        [7,0],[7,1],[7,2],
        [8,0],[8,1],[8,2],
        ])
    e = e1 + e2
    d.constrain(e)

    print 'there are that many constraints ', len(e)
    print 'there are that many DIs ', len(d.u)
    print 'that leaves that many independent DIs ', len(d.u2)-len(e)
    #the last six coordinates in u correspond to the pure cell changes
    coords = PC(d.x_ref.flatten(),d.masses,unit=1.0,atoms=d.atoms,ic=d.ic, Li=d.get_constrained_U(),
            biArgs={'RIIS': True, 'maxiter': 900, 'eps': 1e-6, 'maxEps':1e-6, 
                'col_constraints': d.constraints_cart,
                #'row_constraints':d.constraints_int,
                #needed for sparseBackIteration...empty columns or rows make cholesky very messy
                #'iclambda': 1e-6,
                })
    
    coords.write_jmol('cart_constrained_Pd2x2x4.jmol')

    print 'X0  ', coords.x
    print 'c0 ', coords.cell
    coords.s[len(e):]= 10.0
    coords.getX()
    print 'X1  ', coords.x
    print 'c1 ', coords.cell
