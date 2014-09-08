#from ase.lattice.surface import fcc111, add_adsorbate
#from ase.calculators.dftdisp import dftdisp
from ase import Atoms
from ase.calculators.dftdisp import dftdisp
from ase.all import *
from ase.calculators.dftb import Dftb

from ase.calculators.qmme import qmme
from ase.optimize import BFGS
from INTERNALS.globaloptimization.betterhopping import BetterHopping
import numpy as np

adsorbate = '../testsystems/NH3_on_Ag100.traj'
d='/home/konstantin/software/DFTB/param/SURF/'
molecule = read(adsorbate)

mixer = {'name': 'Pulay', 'convergence': 1E-7}

slab=molecule[0:8]

calc_dftb = Dftb(label='dftb',atoms=molecule,kpts=(2,2,1),
                 Hamiltonian_MaxAngularMomentum_ = '',
                 Hamiltonian_MaxAngularMomentum_N = '"p"',
                 Hamiltonian_MaxAngularMomentum_H = '"s"',
                 Hamiltonian_MaxAngularMomentum_Ag = '"d"',
                 Hamiltonian_MaxSCCIterations = 2000,
                 Hamiltonian_SCC ='YES',
                 Hamiltonian_SCCTolerance = 1.0E-007,
                 Hamiltonian_SlaterKosterFiles_Prefix=d,
                 )

calc_vdw = dftdisp(atoms=None,
                   sedc_print_level=5,
                   sedc_pbc_g_only_intra=[0, -1],
                   sedc_scheme='TS-SURF',
                   sedc_n_groups=2,
                   sedc_groups=[8,3],
                   sedc_pbc_g_switches=[[0, 0, 0, 0, 0, 0],
                                        [1, 1, 0, 1, 1, 0]],
                   sedc_tssurf_vfree_div_vbulk=[1.00, 1.00, 1.00],
                   sedc_do_standalone=False)

QMMM = qmme(atoms=molecule,
            nqm_regions=1,
            nmm_regions=1,
            qm_calculators=[calc_dftb],
            mm_calculators=[calc_vdw],
            qm_atoms=[[(4,12)]],
            mm_mode='allatoms')

molecule.set_calculator(QMMM)

bh = BetterHopping(atoms=molecule,
                  temperature=100 * kB,
                  dr=1.5,
                  optimizer=BFGS,
                  fmax=2,
                  logfile='tt.log',
                  maxmoves=50,
                  movemode=1,
                  numdelocmodes=14,
                  adsorbmask=(8,11)
                  )
bh.run(50)


# Einkommentieren fuer eine Geometrieoptimierung
#f = FixAtoms(mask=[atom.symbol == 'Au' for atom in atoms])
#atoms.set_constraint(f)
#dyn = BFGS(atoms, trajectory='ReA_Au.traj')
#dyn.run(fmax=0.025)
