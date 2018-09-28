from ase.all import *
from ase.constraints import FixAtoms
from winak.globaloptimization.betterhopping import BetterHopping

#read in geometry
atoms = read('less.traj') #mol+surface
atoms.set_pbc([True,True,False])

#dont forget to set DFTB_COMMAND
slako_dir = '/user/panosetti/data/progs/DFTB/parameters/pbc/pbc-0-3/'

calc = Dftb(label='dftb',
    kpts = (3,3,1),
    atoms=atoms,
    Hamiltonian_SCC ='YES',
    Hamiltonian_Charge = 0,
    Hamiltonian_MaxSCCIterations = 100,
    Hamiltonian_MaxAngularMomentum_ = '',
    Hamiltonian_MaxAngularMomentum_Si = '"p"',
    Hamiltonian_MaxAngularMomentum_H = '"s"',
    Hamiltonian_SlaterKosterFiles_='Type2FileNames',
    Hamiltonian_SlaterKosterFiles_Prefix=slako_dir,
    Hamiltonian_SlaterKosterFiles_Separator='"-"',
    Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
    Hamiltonian_Filling_ = 'Fermi',
    Hamiltonian_Filling_Temperature = 0.001,
#    Hamiltonian_EwaldParameter = 0.1,
    )

atoms.set_calculator(calc)
c = FixAtoms(mask=[atom.tag > 3 for atom in atoms])  ### MIND make this consistent; other way around in unrec
atoms.set_constraint(c)
cell=atoms.get_cell()

dyn=BFGS(atoms,trajectory='relax.traj')
dyn.run(fmax=0.01)

