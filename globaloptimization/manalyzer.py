# winak.globaloptimization.manalyzer
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

from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.globaloptimization.disspotter import *
from ase.io.trajectory import Trajectory
import numpy as np

class Manalyzer:
    """
    uses RMSD to determine which minma are the same.
    """
    def __init__(self,trajectory,rmsd=-1,lenthr=-1,edif=-1,removedis=True,mask=None):
        """
        trajectory can either be a string or a ase.io.trajectory Trajectory type
        removedis: should dissociated minima be deleted?
        """
        if isinstance(trajectory,basestring):
            self.allmin=Trajectory(trajectory,'r')
        else:
            self.allmin=trajectory
        if mask is None:
            self.mask=[0,len(self.allmin[0])]
        else:
            self.mask=mask
        if rmsd==-1:
            self.rmsd=1.2
        else:
            self.rmsd=rmsd
        if lenthr==-1:
            self.lent=2
        else:
            self.lent=lenthr#
        if edif==-1:
            self.edif=0.25
        else:
            self.edif=edif
        self.notdis=[]
        self.dis=[]
        self.min=[]
        self.conf=[]
        
        print 'Calculating Energies...'
        tmp=0
        for i in self.allmin:
            e=DisSpotter(atoms=i[self.mask[0]:self.mask[1]])
            if removedis and e.spot_dis():
                self.dis.append(Minimum(i,e.vcg,e.iclist,e.ic,tmp))
            else:
                self.notdis.append(Minimum(i,e.vcg,e.iclist,e.ic,tmp,i.get_potential_energy()))
            tmp+=1
            print str(tmp)+' done'
        print str(len(self.dis))+' dissociated minima, '+str(len(self.notdis))+ ' undissociated minima, '+str(len(self.allmin))+' total'

    def findConformers(self,strref=None,icsref=None):
        """
        all stretches have to be the same;
        """
        if strref is None or icsref is None:
            stre=self.notdis[0].ic.getStretchBendTorsOop()[0][0]
            ics=self.notdis[0].ic.getStretchBendTorsOop()[1]
        else:
            stre=strref
            ics=icsref
        xx=Trajectory('confs.traj','a')
        yy=Trajectory('notconfs.traj','a')
        cnt=0
        for i in self.notdis:
            istre=i.ic.getStretchBendTorsOop()[0][0]
            iics=i.ic.getStretchBendTorsOop()[1]
            same=True
            if len(stre)>len(istre):
                same=False
            else:
                for j in stre:
                    #print j
                    #print i.oldn
                    if not (iics[j][0]==ics[j][0] and iics[j][1]==iics[j][1]):
                        same=False
                        break
            if same:
                xx.write(i.atoms)
                cnt+=1
                print str(cnt)+' - '+str(i.oldn)
                self.conf.append(i)
            else:
                yy.write(i.atoms)
            

"""
THIS FUNCTION IS UNTESTED
    def dothebartman(self,onlyconf=False):
        if onlyconf:
            self.findConformers()
            iter=self.conf
            pre='c'
        else:
            iter=self.notdis
            pre=''
        for i in iter:
            if len(self.min)==0:
                tmpmin=[]
                tmpmin.append(i)
                self.min.append(tmpmin)
            else:
                found=False
                for j in self.min:
                    tmpic=icSystem(j[0].iclist,len(j[0].atoms),masses=j[0].atoms.get_masses(),xyz=i.atoms.get_positions().flatten())
                    cic=tmpic()
                    tmpsum=0
                    for k in range(len(cic)):
                        tmpsum+=(cic[k]-j[0].cic[k])**2
                    rmsd=np.sqrt(tmpsum/len(cic))
                    print 'rmsd from '+str(i.oldn)+ ' to '+str(j[0].oldn)+' = '+str(rmsd)+' ,dif= '+str(np.abs(len(i.cic)-len(j[0].cic)))+' , Edif = '+str(np.abs(i.E-j[0].E))
                    if rmsd<self.rmsd and np.abs(len(i.cic)-len(j[0].cic))<self.lent and np.abs(i.E-j[0].E)<self.edif:
                        j.append(i)
                        found=True
                        break
                if not found:
                    tmpmin=[]
                    tmpmin.append(i)
                    self.min.append(tmpmin)

        for i in range(len(self.min)):
            xx=Trajectory(pre+str(i)+'.traj','w')
            print 'minim: '+str(i)
            for j in range(len(self.min[i])):
                xx.write(self.min[i][j].atoms)
                print self.min[i][j].oldn
"""
class Minimum:
    """
    convenience class, used to avoid a million lists
    """
    def __init__(self,atoms,vcg,icl,ic,oldnum,en=None):
        self.atoms=atoms
        self.vcg=vcg
        self.iclist=icl
        self.ic=ic
        self.oldn=oldnum
        self.cic=ic()
        self.E=en
