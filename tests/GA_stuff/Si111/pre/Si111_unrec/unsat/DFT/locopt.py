from ase.all import *
from ase.constraints import FixAtoms
from winak.globaloptimization.betterhopping import BetterHopping

'''
### AIMS local
import os
import subprocess
import sys
os.environ['OMP_NUM_THREADS'] = "1"
aims_binary = 'mpirun.openmpi -n 8 /usr/local/stow/fhi-aims.020714/bin/aims.020714.static.scalapack.mpi.x > aims.out'
'''

aims_defaults = '/data/panosetti/progs/FHI-aims/fhi-aims.081912/species_defaults/light_194'
aims_binary = 'mpirun.openmpi --hostfile $PBS_NODEFILE /usr/local/stow/fhi-aims.020714/bin/aims.020714.static.scalapack.mpi.x > aims.out'

#read in geometry
atoms = read('less.traj') #mol+surface
atoms.set_pbc([True,True,False])

calc = Aims(xc='pbe',
            k_grid="1 1 1",
            spin='none',
            charge=0.0,
            relativistic ='atomic_zora scalar 1e-9',
            sc_accuracy_etot=1e-5, #6,4,5
            sc_accuracy_eev=1e-2,
            sc_accuracy_rho=1e-4,
            sc_accuracy_forces=1e-4, #4
            species_dir=aims_defaults,
            command=aims_binary,
            charge_mix_param=0.5,
            occupation_type='gaussian 0.1',
            ##large system settings
            empty_states=3,
            use_local_index='.true.',
            load_balancing='.true.',
            density_update_method='density_matrix',
            collect_eigenvectors='.false.'
            )

atoms.set_calculator(calc)
c = FixAtoms(mask=[atom.tag > 3 for atom in atoms])
atoms.set_constraint(c)
cell=atoms.get_cell()

dyn=BFGS(atoms,trajectory='relax.traj')
dyn.run(fmax=0.1)
