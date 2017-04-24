# winak.globaloptimization.disspotter
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

from winak.curvilinear.InternalCoordinates import icSystem
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.Coordinates import InternalCoordinates as IC
import numpy as np


class DisSpotter:
    """
    Tells you if a molecule is dissociated or not.
    """
    def __init__(self, atoms):
        if isinstance(atoms,Atoms):
            self.molecule=atoms
        else:
            self.molecule=read(atoms)
        self.visited=[False]*(len(self.molecule)+1)
        self.vcg=VCG(self.molecule.get_chemical_symbols(),masses=self.molecule.get_masses())
        self.iclist=self.vcg(self.molecule.get_positions().flatten())
        self.ic=icSystem(self.iclist,len(self.molecule),masses=self.molecule.get_masses(),xyz=self.molecule.get_positions().flatten())

        self.stre=self.ic.getStretchBendTorsOop()[0][0]
        self.ics=self.ic.getStretchBendTorsOop()[1]

    def numConnections(self,j):
        """ics: as in the main program bottom
           stre: list of stretches
           j: atom you want to find
        """
        found=0
        s=[]
        for i in self.stre:
            a=self.ics[i]
            if a[0]==j or a[1]==j:
                found+=1
                s.append(i)
        return found,s

    def fragment(self,j):
        """
        This might seem complicated, but it is actually not.
        """
        ret=0
        if not self.visited[j]:
            ret+=1
            self.visited[j]=True
            x=self.numConnections(j)
            #print 'looking at '+str(j)
            #print str(j)+' has '+str(x[0])+' connections'
            if x[0]==1:
                #print 'ending in atom '+str(j)+' and returning '+str(ret)
                return ret
            else:
                for l in x[1]:
                    tmp=self.ics[l][1]
                    if tmp==j:
                        tmp=self.ics[l][0]
                    #print str(j)+' is connected to atom '+str(tmp)
                    ret+=self.fragment(tmp)
                    #print 'back at atom '+str(j)
                    #print 'ret is now '+str(ret)
            return ret
        else:
            #print 'Atom '+str(j)+' already accounted for'
            return 0
        
    def get_fragments(self):
        """
        Returns all the fragments
        """
        self.visited=[False]*(len(self.molecule)+1)
        self.visited[0]=True #whoever decided to make ics start at 1, go to hell
        ret=[]
        while True:
            try:
                a=self.visited.index(False)
                tmp=[]
                tmp=self.get_fragment(a)
                ret.append(tmp)
            except ValueError:
                break
        return ret
        
        
    def get_fragment(self,j):
        """
        This might seem complicated, but it is actually not. 
        """
        ret=[]
        if not self.visited[j]:
            ret.append(j)
            self.visited[j]=True
            x=self.numConnections(j)
            #print 'looking at '+str(j)
            #print str(j)+' has '+str(x[0])+' connections'
            if x[0]==1 and self.visited[self.ics[self.stre[x[1]]][0]] and self.visited[self.ics[self.stre[x[1]]][1]]:
                #print 'ending in atom '+str(j)+' and returning '+str(ret)
                return ret
            else:
                for l in x[1]:
                    tmp=self.ics[l][1]
                    if tmp==j:
                        tmp=self.ics[l][0]
                    #print str(j)+' is connected to atom '+str(tmp)
                    r2=self.get_fragment(tmp)
                    for q in r2:
                        ret.append(q)
                    #print 'back at atom '+str(j)
                    #print 'ret is now '+str(ret)
            return ret
        else:
            #print 'Atom '+str(j)+' already accounted for'
            return []
        
    def spot_dis(self):
        """
        Returns true if dissociation happened
        """
        return len(self.molecule)>self.fragment(1)#1 is arbitrary
