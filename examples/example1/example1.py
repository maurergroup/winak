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

from ase.io import read
from ase.optimize.bfgs import BFGS
from ase.calculators.emt import EMT
from winak.globaloptimization.betterhopping import BetterHopping
from winak.constants import kB

atoms = read('Pd-cluster.xyz')
atoms.set_calculator(EMT())


bh = BetterHopping(atoms=atoms, temperature=100*kB,
        dr = 1.1,
        optimizer=BFGS,
        fmax=0.025,
        logfile='global.log',
        movemode=1,
        )

bh.run(20)

