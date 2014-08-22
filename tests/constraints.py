from ase.all import *
from INTERNALS.globaloptimization.delocalizer import *
from INTERNALS.curvilinear.Coordinates import DelocalizedCoordinates as DC
m=read('clethen.xyz')
e=Delocalizer(m)
d=[]

n=17

ce=np.zeros(n)
ce[0]=1
d.append(ce)
ce=np.zeros(n)
ce[1]=1
d.append(ce)
ce=np.zeros(n)
ce[2]=1
d.append(ce)
ce=np.zeros(n)
ce[3]=1
d.append(ce)
ce=np.zeros(n)
ce[4]=1
d.append(ce)


e.constrain(d)

coords=DC(e.x_ref.flatten(), e.masses, internal=True, atoms=e.atoms, \
             ic=e.ic, L=None, Li=None,u=e.u2)

coords.write_jmol('cdol') #constrained delocalizing out loud

"""
The next loop prints the C-C distance after displacing along a delocalized internal.
It varies by about 10%.
"""

tmp=read('clethen.xyz')
for i in coords.get_vectors():
    tmp.set_positions(e.x_ref+i)
    #view(m)
    print np.linalg.norm(tmp[1].get_position()-tmp[0].get_position())

