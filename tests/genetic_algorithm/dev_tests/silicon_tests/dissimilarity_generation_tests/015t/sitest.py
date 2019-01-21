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

#### deletes the results of the last execution
if os.path.exists("Results.traj"):
    os.remove("Results.traj")
if os.path.exists("GS.log"):
    os.remove("GS.log")




stru = read('/data/guest20/Executables/winak/tests/GA_stuff/Si111/Si111_3x3/less.traj')

pop = [stru]

#### set constraints.
#### 999 = free
#### 1 = not involved in mutations and matings.
#### 2 = not involved in mutation and matings, and fixed in relaxations.
for stru in pop:
    for atom in stru:
        if atom.z < 9 and atom.z > 6:
            atom.tag = 1
        else:
            atom.tag = 999 
        if atom.z < 6:
            atom.tag = 2




################ Controls ####################
################ Displacement ################
Xparameter = 4 

MatingManager = "FabioMating"
MutationManager = "FabioMutation"

MatingOperator = "SinusoidalCut"   
MatingOperatorParameters = {"FixedElements":[],"NumberOfAttempts":50,"collision_threshold":0.001}  ####collision_threshold should be chosen analyzing the behavior of the optimizer

MutationOperator = "GC"
FixedElements_GC = ["H"]
MutationOperatorParameters = {"prob":0.0,"stepwidth":2.0,"periodic":True,"constrain":False,"GA_mode":True,"adsorbate":999,"atm":"Si","adjust_cm":False}

##############################################
############### Evaluation ###################

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
EE = "potEE"

EEparameters = {"calc":calc_dftb,"opt":BFGS,"fmax":1.0,"optlog":"local_optimization.log"}
similarity_threshold = 0.999

##############################################
############### Criterion ####################
popsize = 8    

##############################################
############### Screening ####################
break_limit = 30
break_limit_top = 100
savegens = True
number_of_generations = 10 
##############################################

MatingParameters = {"MatingOperator":MatingOperator,
                    "MatingOperatorParameters":MatingOperatorParameters
                   } 
MutationParameters = {"MutationOperator":MutationOperator,
                      "FixedElements_GC":FixedElements_GC,
                      "MutationOperatorParameters":MutationOperatorParameters
                     }
DisplacementParameters = {"MatingManager":MatingManager,
                          "MatingParameters":MatingParameters,
                          "MutationManager":MutationManager,
                          "MutationParameters":MutationParameters,
                          "Xparameter":Xparameter
                         }




GeneticDisplacer = FabioManager(**DisplacementParameters)
PopulationEvaluator = FabioPopEvaluator(EE,EEparameters,similarity_threshold)
Criterion = FabioSelection(popsize)


Screener = GeneticScreener(PopulationEvaluator,GeneticDisplacer,Criterion,savegens=savegens, break_limit=break_limit, break_limit_top = break_limit_top)
starting_population = Screener.generate(pop,popsize,minimum_diversity=0.015)
write("starting_population.traj",starting_population)
#Screener.run(starting_population,number_of_generations)


