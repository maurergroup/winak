from winak.globaloptimization.manalyzer import *
from ase.io.trajectory import *

p=Trajectory('../testsystems/min_c.traj','r')
m=Manalyzer(p,removedis=False,mask=[16,len(p[0])])
m.findConformers()
                                    
