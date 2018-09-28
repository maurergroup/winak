#winak.SOAP_interface
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
import os
import numpy as np
from ase.io import read
from ase.io import write
import subprocess 
from datetime import datetime

def compare(a,b):
    """ Compares structures and returns their similarity as a float 0.0 <--> 1.0, with 1.0 = identical structures """
    pair = [a,b]
    #random= np.random.randint(99999)
    #name = "pair"+str(random)+".traj"
    #write(name,pair)

    write("pair.xyz",pair)
    with open('compare.log','w') as out:
        subprocess.call("/data/panosetti/shared/.venvs/stretch/glosim/glosim.py pair.xyz -n 9 -l 9 -g 0.3 -c 3 --zeta 2 --kernel match",stdout=out,shell=True)
        
    matrix = np.genfromtxt("pair-n9-l9-c3.0-g0.3_match.k",skip_header=1)
    result = matrix[0,1]
    
    return result

def quantify_dissimilarity(pop):
    """Receives a population of structures and returns the average DISsimilarity between the structures as a float 0.0 <--> 1.0 """
    write("population.xyz",pop)
    with open('quantify_dissimilarity.log','w') as out:
       subprocess.call("/data/panosetti/shared/.venvs/stretch/glosim/glosim.py population.xyz -n 9 -l 9 -g 0.3 -c 6  --zeta 2 --kernel match",stdout=out,shell=True)
    matrix = np.genfromtxt("population-n9-l9-c6.0-g0.3_match.k",skip_header=1)
    length = len(matrix)
    higher_tri = matrix[np.triu_indices(length,k=1)]
    vector_length = len(higher_tri)
    summation = sum(higher_tri)
    dissimilarity = (vector_length - summation)/vector_length
    return(dissimilarity)

def sim_matrix(pop):
    """Receives a population of structures and returns the relative kernel matrix"""
    time = datetime.now()
    name = str(time.microsecond)
    write(name+".xyz",pop)
    repeat= True
    count = 0
    while repeat:
        repeat = False
        count += 1
        print("ATTEMPT NUMBER ",count)
        try:
            with open('sim_matrix.log','w') as out:
                subprocess.call("/data/panosetti/shared/.venvs/stretch/glosim/glosim.py "+name+".xyz -n 9 -l 9 -g 0.3 -c 6  --zeta 2 --kernel match",stdout=out,shell=True)
            matrix = np.genfromtxt(name+"-n9-l9-c6.0-g0.3_match.k",skip_header=1)
        except:
            repeat = True
    return(matrix)
