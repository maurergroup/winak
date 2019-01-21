from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager

import numpy as np

def get_sorted_dist_list(atoms, mic=False):
    """ Utility method used to calculate the sorted distance list
        describing the cluster in atoms. """
    numbers = atoms.numbers    
    print("numbers:",numbers)
    unique_types = set(numbers)
    print("unique:",unique_types)
    pair_cor = dict()
    for n in unique_types:
        i_un = [i for i in range(len(atoms)) if atoms[i].number == n]
        print("i_un",i_un)
        d = []
        for i, n1 in enumerate(i_un):
            for n2 in i_un[i + 1:]:
                d.append(atoms.get_distance(n1, n2, mic))
        print("d non sorted",d)
        d.sort()
        print("d sorted",d)
        pair_cor[n] = np.array(d)
    return pair_cor

pop = Trajectory('Results.traj','r')

stru1 = Atoms('C3O3',positions=[(0,0,0),(2,0,0),(2,3,0),(0,1,0),(0,2,0),(1,0,0)]) 
stru2 = pop[1]

sorted1 = get_sorted_dist_list(stru1)
print("SORTED:")
print(sorted1)
view(stru1)
