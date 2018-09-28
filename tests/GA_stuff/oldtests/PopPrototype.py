from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
import numpy as np

pop = [] 

materials=['Ni','Co','Zn','Cu']


for el in materials:
    info=dict()
    info['fitness']=np.random.rand()
    stru = Atoms(str(4)+el,positions=[(0,0,0),(0,3,0),(3,0,0),(3,3,0)],cell=[6,6,3],info=info)
    stru = stru.repeat((3,3,2))
    pop.append(stru.copy())

write('pop.traj',pop)

result = Trajectory('pop.traj','r')

view(result)


stru = result[0]
num = np.random.randint(0,len(stru))
newel = np.random.choice(materials)

stru[num].symbol = newel

view(stru)





# toslice=[]
# 
# for stru in result:
#     toslice.append(stru.copy())
# 
# 
# for stru in toslice:
#     print('total', stru.info)
# 
# for stru in toslice:
#     candidates = list(toslice)
#     candidates.remove(stru)
#     for stra in candidates:
#         print('candidates',stra.info)

#for stru in toslice[2:]:
#    print('end',stru.info)

#for cont in range(5):
 #   x = np.random.randint(0,len(toslice))
  #  print(x)
   # print(toslice[x].info)   

#view(result)
