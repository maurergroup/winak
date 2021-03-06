from ase.all import *
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
m=read('../testsystems/clethen.xyz')
e=Delocalizer(m)
d=[]

n=17

e.constrainStretches()

coords=DC(e.x_ref.flatten(), e.masses, internal=True, atoms=e.atoms, \
             ic=e.ic, L=None, Li=None,u=e.get_U())

coords.write_jmol('cdol') #constrained delocalizing out loud

"""
The next loop prints the C-C distance after displacing along a delocalized internal.
It varies by about 10%.
"""

tmp=read('../testsystems/clethen.xyz')
for i in coords.get_vectors():
    tmp.set_positions(e.x_ref+i)
    #view(m)
    print np.linalg.norm(tmp[1].get('position')-tmp[0].get('position'))

