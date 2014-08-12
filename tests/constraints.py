from ase.all import *
from INTERNALS.globaloptimization.delocalizer import *
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

vc=e.vc
x=e.x_ref

e.write_jmol('asdf.xyz',True)

#tmp=read('/home/konstantin/Documents/molecules/thctk_ex/clethen.xyz')
#for i in vc:
#    tmp.set_positions(x+i)
#    #view(m)
#    print np.linalg.norm(tmp[1].get_position()-tmp[4].get_position())

