from ase.all import *
from collections import Counter
import numpy as np

class Stoichiometry:
    """ Keeps track of changes in stoichiometry. Also spawns new trajectories when the composition changes."""
    def __init__(self):
        #"""stoich is the stoichiometry as a dictionary. comp is an array with the number of atoms per atom type. MUST be intialized with .get()"""
        ## CP there's probably better ways to do this but it's just the first that came to mind
        self.stoich={}
        self.comp=np.asarray(self.stoich.values())
        
    def get(self,atoms):
        stoich=dict(Counter(atoms.get_chemical_symbols()))
        self.stoich=stoich
        self.comp=np.asarray(stoich.values()) 
        self.formula=''.join("{!s}{!r}".format(symbol,coeff) for (symbol,coeff) in self.stoich.iteritems())
        return stoich #not necessary but it can be useful to have the stoichiometry returned

    def has_changed(self,atoms,fallbackatoms):
        old=self.get(fallbackatoms)
        new=self.get(atoms)
        return not new==old

    def make_traj(self):
        traj = Trajectory('minima_'+self.formula+'.traj', 'a')
        return traj

