from ase.all import *
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
m=read('../testsystems/rea.xyz')
e=Delocalizer(m, dense=True)
d=[]

n=17

c_vec = e.constrainStretches()
e.constrain(c_vec)
coords=DC(e.x_ref.flatten(), e.masses, internal=True, atoms=e.atoms, \
             ic=e.ic, L=None, Li=None,u=e.get_constrained_U(),
             biArgs={'maxiter':100,'RIIS': True, 
                 #'col_constraints':e.constraints_cart,
                 #'row_constraints':e.constraints_int,
                 })

coords.write_jmol('cdol') #constrained delocalizing out loud

"""
The next loop prints the C-C distance after displacing along a delocalized internal.
It varies by about 10%.
"""

#tmp=read('../testsystems/clethen.xyz')
#for i in coords.get_vectors():
    #tmp.set_positions(e.x_ref+i)
    ##view(m)
    #print np.linalg.norm(tmp[1].get_position()-tmp[0].get_position())

