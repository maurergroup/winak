from ase.all import *
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.InternalCoordinates import icSystem
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import Set_of_CDCs

system = read('../testsystems/h2_on_Pd100.traj')
print system.get_masses()
print len(system.get_masses())
print system.positions
#subsystem1  H2 molecule
subsystem1=system[:2]
d = Delocalizer(subsystem1)
cell = subsystem1.get_cell()#*[1.0,1.0,1.0]

coords_s1 = CDC(d.x_ref.flatten(),d.masses,unit=1.0,
        atoms=d.atoms,ic=d.ic,u=d.u,cell=cell)

coords_s1.write_jmol('s1.jmol')




#subsystem2 Surface
subsystem2=system[2:]
d = Delocalizer(subsystem2)
cell = subsystem1.get_cell()#*[1.0,1.0,1.0]

coords_s2 = DC(d.x_ref.flatten(),d.masses,unit=1.0,
        atoms=d.atoms,ic=d.ic,u=d.u)
        #atoms=d.atoms,ic=d.ic,u=d.u,cell=cell)

coords_s2.write_jmol('s2.jmol')


#put together

coords = Set_of_CDCs([coords_s1, coords_s2])
print coords.masses
print len(coords.masses)
print coords.x0.reshape(-1,3)

s = coords.getS(coords.x0)
print s
x= coords.getX(s)

print x.reshape(-1,3)
