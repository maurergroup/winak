from INTERNALS.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from INTERNALS.globaloptimization.disspotter import *
from ase.io.trajectory import PickleTrajectory
import numpy as np
from hotbit import *
from hotbit.coulomb import MultipoleExpansion

class Manalyzer:
    """
    uses RMSD to determine which minma are the same.
    """
    def __init__(self,trajectory,rmsd=-1,lenthr=-1,edif=-1):
        """
        trajectory can either be a string or a ase.io.trajectory PickleTrajectory type
        """
        if isinstance(trajectory,basestring):
            self.allmin=PickleTrajectory(trajectory,'r')
        else:
            self.allmin=trajectory
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

        d = '/home/konstantin/software/HOTBIT/param/SURF/'
        elm = {'H': d + 'H.elm',
               'C': d + 'C.elm',
               'N': d + 'N.elm',
               'O': d + 'O.elm'}
        tab = {'CH': d + 'C_H.par',
               'HH': d + 'H_H.par',
               'CC': d + 'C_C.par',
               'CN': d + 'C_N.par',
               'NH': d + 'N_H.par',
               'NN': d + 'N_N.par',
               'rest': 'default'}
        mixer = {'name': 'Pulay', 'convergence': 1E-7}
        calc_dftb = Hotbit(txt='hotbit.txt',
                   elements=elm,
                   mixer=mixer,
                   tables=tab,
                   SCC=True,
                   verbose_SCC=True,
                   width=0.10,
                   maxiter=100,
                   coulomb_solver=MultipoleExpansion(
                       n=(3, 3, 1),
                       k=(1, 1, 1)),
                   kpts=(1,1,1)
                   )

        print 'Calculating Energies...'
        tmp=0
        for i in self.allmin:
            e=DisSpotter(atoms=i)
            if e.spot_dis():
                self.dis.append(Minimum(i,e.vcg,e.iclist,e.ic,tmp))
            else:
                i.set_calculator(calc_dftb)
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
        xx=PickleTrajectory('confs.traj','a')
        for i in self.notdis:
            istre=i.ic.getStretchBendTorsOop()[0][0]
            iics=i.ic.getStretchBendTorsOop()[1]
            same=True
            for j in stre:
                if not (iics[j][0]==ics[j][0] and iics[j][1]==iics[j][1]):
                    same=False
                    break
            if same:
                xx.write(i.atoms)
                print i.oldn
                self.conf.append(i)


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
            xx=PickleTrajectory(pre+str(i)+'.traj','w')
            print 'minim: '+str(i)
            for j in range(len(self.min[i])):
                xx.write(self.min[i][j].atoms)
                print self.min[i][j].oldn

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
