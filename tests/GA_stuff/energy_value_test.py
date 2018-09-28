from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import FabioManager
from winak.screening.energyevaluator import FabioPopEvaluator
from winak.screening.criterion import FabioSelection
from winak.screening.ultimatescreener import GeneticScreener
from ase.calculators.dftb import Dftb
import os

from ase.optimize import BFGS
from ase.calculators.morse import MorsePotential

from ase.calculators.emt import EMT

import numpy as np


#### DFTB settings ####

dftb_binary='mpirun.openmpi --np 8 /data/panosetti/shared/.venvs/stretch/bin/dftb+ > dftb.out'
slako_dir = '/user/panosetti/data/progs/DFTB/parameters/pbc/pbc-0-3/'

calc_dftb = Dftb(label='dftb',
        command = dftb_binary,
        kpts=(3, 3, 1),
        Hamiltonian_MaxAngularMomentum_='',
        Hamiltonian_MaxAngularMomentum_H = 's',
        Hamiltonian_MaxAngularMomentum_Si = 'p',
        Hamiltonian_SCC = 'YES',
        Hamiltonian_SlaterKosterFiles_Prefix=slako_dir,
        Hamiltonian_Filling_ = 'Fermi',
        Hamiltonian_Filling_Temperature = 0.01,
        Hamiltonian_EwaldParameter = 0.5
)
#######################





structure = read('/data/guest20/Executables/winak/tests/GA_stuff/Si111/Si111_3x3/less.traj'    )
structure.set_calculator(calc_dftb)
energia_iniziale = structure.get_potential_energy()
struttura_iniziale = structure.copy()
optimizer = BFGS(structure,trajectory="test_energie.traj")
optimizer.run(fmax=0.1)
energia_finale = structure.get_potential_energy()

print("start: ",energia_iniziale,"\n"+"end: ",energia_finale)

risultati = [struttura_iniziale,structure]
write('risultati.traj',risultati)




#EEparameters = {"calc":calc_dftb,"opt":BFGS,"fmax":1.0,"optlog":"local_optimization.log"}
#similarity_threshold = 0.999


