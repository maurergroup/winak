# winak.screening.composition
#
#    winak - python package for structure search and more in curvilinear coordinates
#    Copyright (C) 2016  Reinhard J. Maurer and Konstantin Krautgasser 
#    
#    This file is part of winak 
#        
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>#

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

