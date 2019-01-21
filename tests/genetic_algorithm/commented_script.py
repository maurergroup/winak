from ase import Atoms
from ase.io import Trajectory
from ase.io import read
from ase.io import write
from ase.visualize import view
from winak.screening.displacer import MainManager
from winak.screening.energyevaluator import BasicPopEvaluator
from winak.screening.criterion import FlexibleSelection
from winak.screening.ultimatescreener import GeneticScreener
from ase.calculators.dftb import Dftb
import os
from ase.optimize import BFGS
from ase.calculators.morse import MorsePotential
from ase.calculators.emt import EMT
import numpy as np

#### Performs a global optimization on a given structure through a genetic algorithm ####

#### deletes the results of the last execution
if os.path.exists("Results.traj"):
    os.remove("Results.traj")
if os.path.exists("GS.log"):
    os.remove("GS.log")
if os.path.exists("numerical_log.xlsx"):
    os.remove("numerical_log.xlsx")
    
#### input structure(s)
stru = read('/data/guest20/Executables/winak/tests/GA_stuff/Si111/Si111_3x3/less.traj') #path to an ase .traj file. If the.traj includes multiple structure, a number must be included

pop = [stru] #if a single structure is used, it doesn't have to be given as a list

#### Impose constraints on the atoms of the input structure, depending on their .x , .y and .z coordinates
#### 999 = the atom is free
#### 1 = the atom is not involved in mutations and matings, but is free to relax during local optimizations
#### 2 = the atom is not involved in mutation and matings, and is fixed in relaxations
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
Xparameter = 4 #Defines the mating/mutation ratio. See screening.displacerr

MatingManager = "SemiRandomMating" #See screening.displacer. New mating managers can be developed and implemented
MutationManager = "ComplementaryMutation" #See screening.displacer. New mutation managers can be developed and implemented

MatingOperator = "SinusoidalCut" #See screening.displacer
MatingOperatorParameters = {"FixedElements":[], #The number of atoms of these chemical elements will not change during matings
                            "NumberOfAttempts":50, #maximum number of attempts before giving up on a specific mating
                            "collision_threshold":0.001}  #The threshold under which two atoms are considering as colliding, expressed as a fraction of the sum of the atomic radii of the two atoms. Should be chosen analyzing the behavior of the optimizer

MutationOperator = "GC" #See screening.displacer
FixedElements_GC = ["H"] #The number of atoms of these chemical elements will not change during mutations
MutationOperatorParameters = {"prob":0.0, #see screening.displacer
                              "stepwidth":2.0,#see screening.displacer
                              "periodic":True,#must be True if pbc are applied
                              "constrain":False,#see screening.displacer 
                              "GA_mode":True,#must be True
                              "adsorbate":999,#applies the constraints we have previously selected
                              "atm":"Si",#selectes atoms allowed to change. redundand with fixedelements_gc
                              "adjust_cm":False}#see screening.displacer

##############################################
############### Evaluation ###################

#### DFTB settings ####

#see dftb+ documentation for details
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
EE = "potEE" #fitness = stability

EEparameters = {"calc":calc_dftb, #select calculator, local opt. method, and related parameters
                "opt":BFGS,
                "fmax":1.0,
                "optlog":"local_optimization.log"}
similarity_threshold = 0.999 #select the SOAP similarity index indicating two structures are identical

##############################################
############### Criterion ####################
popsize = 8 #size of the population of structures
fitness_preponderance = 1 #relative importance of average fitness over diversity

#######################################################
############### Initial generation ####################
minimum_diversity = 0.035 #the minimum diversity index for the first generation

##############################################
############### Screening ####################
break_limit = 30 #number of generations without any new structure after which the algorithm stops
break_limit_top = 100 #number of generations in which the best structure is the same after which the algorithm stops
savegens = True #if True, stores all generations as .traj files
number_of_generations = 10 #termination condition
##############################################
#packs all the parameters we have so far defined
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




GeneticDisplacer = MainManager(**DisplacementParameters)
PopulationEvaluator = BasicPopEvaluator(EE,EEparameters,similarity_threshold)
Criterion = FlexibleSelection(popsize,fitness_preponderance) #different 'mode's can be selected
Screener = GeneticScreener(PopulationEvaluator,GeneticDisplacer,Criterion,savegens=savegens, break_limit=break_limit, break_limit_top = break_limit_top)

starting_population = Screener.generate(pop,popsize,minimum_diversity=minimum_diversity) #generates the initial population from the input structure(s)
write("starting_population.traj",starting_population)
#Screener.run(starting_population,number_of_generations) #runs the optimization

os.system('echo | mutt -a numerical_log.xlsx -- user@mail.address') #sends an email with a .xlsx numerical log file as an attachment
