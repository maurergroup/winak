# winak.globaloptimization.betterhopping
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

import numpy as np
from datetime import datetime
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory
from winak.curvilinear.Coordinates import DelocalizedCoordinates as DC
from winak.curvilinear.Coordinates import PeriodicCoordinates as PC
from winak.globaloptimization.delocalizer import *
from winak.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from winak.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from winak.curvilinear.InternalCoordinates import icSystem

class BetterHopping(Dynamics):
    """Basin hopping algorythm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)

    adopted by Konstantin Krautgasser, April 2014
    """

    def __init__(self, atoms,
                    temperature=100 * kB,
                    optimizer=FIRE,
                    optimizer2=None, #which optimizer to use to fmax_mult*fmax
                    fmax_mult=15, #at what multiple of fmax do you want to use second optimizer
                    fmax=0.1,
                    dr=0.1,
                    logfile='-',
                    trajectory='lowest.traj',
                    optimizer_logfile='stdout.log',
                    local_minima_trajectory='temp_local_minima.traj',	#local minima found
                    adjust_cm=True,
                    movemode=0,		#Pick a way for displacement. 0->random cartesian, 1->delocalized internals, 2->Periodic
                    maxmoves=1000,	#Should prevent an issue, where you get stuck in a structure, for which get_energy() fails
                    numdelocmodes=1,    #should a LC of modes be applied for the displacement? How many should be combined?
                    adsorbmask=None,	#mask that specifies where the adsorbate is located in the atoms object (list of lowest and highest pos)
                    cell_scale=[1.0,1.0,1.0],    #used for translation in adsorbates
                    constrain=False):   #constrain stretches?
        Dynamics.__init__(self, atoms, logfile, trajectory)

        if adsorbmask is None:
            self.adsorbate=(0,len(atoms))
            self.ads=False
        else:
            self.adsorbate=adsorbmask
            self.ads=True
        self.fmax_mult=fmax_mult
        self.cell_scale=cell_scale
        self.kT = temperature
        if numdelocmodes<1:
            if self.ads:
                self.numdelmodes=int(np.round(numdelocmodes*len(atoms[self.adsorbate[0]:self.adsorbate[1]])*3))#3N dis in adsorbates
            else:
                self.numdelmodes=int(np.round(numdelocmodes*(len(atoms)*3-6)))#3N-6 dis in gas phase
        else:
            self.numdelmodes=int(np.round(numdelocmodes))#round and int just for you trolls out there 
        self.optimizer = optimizer
        if optimizer2 is None:
            self.optimizer2=optimizer
        else:
            self.optimizer2=optimizer2
        self.fmax = fmax
        self.mm=maxmoves
        self.dr = dr
        self.lowstep=0
        self.movemode=movemode
        self.movename='Random Cartesian'
        self.minima=[]
        self.constr=constrain
        if movemode==1:
            self.movename='Delocalized Internals, using %i modes'%self.numdelmodes
        elif movemode==2:
            self.movename='Periodic DI'
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None
        self.lmfile=local_minima_trajectory
        self.optimizer_logfile = optimizer_logfile
        self.lm_trajectory = local_minima_trajectory
        if isinstance(local_minima_trajectory, str):
            self.lm_trajectory = Trajectory(local_minima_trajectory,'a', atoms)
        self.startT = datetime.now()
        self.log(msg='STARTING BASINHOPPING at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+':\n Displacements: '+self.movename+' Stepsize: %.3f fmax: %.3f T: %4.2f\n'%(self.dr,self.fmax,self.kT/kB))
        self.positions = 0.0 * self.atoms.get_positions()
        self.Emin = self.get_energy(self.atoms.get_positions()) or 1.e15
        self.rmin = self.atoms.get_positions()
        self.call_observers()
        self.log(-1, self.Emin, self.Emin)

    def check_distances(self,atoms,min=0.25):
        vcg=VCG(atoms.get_chemical_symbols(),masses=atoms.get_masses())
        iclist=vcg(atoms.get_positions().flatten())
        ic=icSystem(iclist,len(atoms), masses=atoms.get_masses(),xyz=atoms.get_positions().flatten())
        stre=ic.getStretchBendTorsOop()[0][0]
        ics=ic.getStretchBendTorsOop()[1]
        ret=True
        for i in stre:
            a=atoms[ics[i][0]-1].position #For some absurd reason, IC stuff starts at atom number 1, not 0...
            b=atoms[ics[i][1]-1].position
            if np.linalg.norm(a-b)<min:
                ret=False
                break
        return ret

    def get_vectors(self,atoms):
        if self.movemode==2:
            deloc=Delocalizer(atoms,periodic=True,dense=True)
            coords=PC(deloc.x_ref.flatten(), deloc.masses, atoms=deloc.atoms, ic=deloc.ic, Li=deloc.u)
        else:
            deloc=Delocalizer(atoms)
            tu=deloc.u
            if self.constr:
                e = deloc.constrainStretches()
                deloc.constrain(e)
                tu=deloc.u2
            if not self.ads:
                coords=DC(deloc.x_ref.flatten(), deloc.masses, atoms=deloc.atoms, ic=deloc.ic, u=tu)
            else:
                cell = self.atoms.get_cell()*self.cell_scale
                coords=CDC(deloc.x_ref.flatten(), deloc.masses, unit=1.0, atoms=deloc.atoms, ic=deloc.ic, u=tu,cell=cell)
        return coords.get_vectors()

    def run(self, steps):
        """Hop the basins for defined number of steps."""
        ro = self.positions
        lastmol=ro.copy()
        Eo = self.get_energy(ro)
        lastworkingpos=ro.copy()
        lastworkinge=Eo
        self.minima.append(self.atoms.copy())
        for step in range(steps):
            En = None
            tries=0
            if self.movemode==1 or self.movemode==2:
                atemp=self.atoms.copy()
                atemp.set_positions(ro)
                try:
                    if not self.ads:
                        vectors=self.get_vectors(atemp)
                    else:
                        vectors=self.get_vectors(atemp[self.adsorbate[0]:self.adsorbate[1]])
                except:
                    #usually the case when the molecule dissociates
                    self.log(msg='      WARNING: Could not create delocalized coordinates. Rolling back!\n')
                    self.atoms.set_positions(lastmol)
                    atemp=self.atoms.copy()
                    if not self.ads:
                        vectors=self.get_vectors(atemp)
                    else:
                        vectors=self.get_vectors(atemp[self.adsorbate[0]:self.adsorbate[1]])
                    ro=lastmol.copy()
                lastmol=ro.copy()
            while En is None:
                if self.movemode==0:
                    rn = self.move(ro)
                elif self.movemode==1 or self.movemode==2:
                    rn=self.move_del(ro,vectors)
                self.atoms.write(str(step)+'.xyz',format='xyz')
                En = self.get_energy(rn)
                tries+=1
                if tries>self.mm:
                    ro=lastworkingpos.copy()	#for some reason, some random steps cause get_energy() failure; seems to be a Hotbit problem
                    Eo=lastworkinge      #we are going to pretend that never happened and reverse last step
                    tries=0
                    self.log(msg='     WARNING: last step caused get_energy() failure; Resetting step\n')
            self.minima.append(self.atoms.copy())
            if En < self.Emin:
                # new minimum found
                self.Emin = En
                self.rmin = self.atoms.get_positions()
                self.call_observers()
                self.lowstep=step
            self.log(step, En, self.Emin)
            lastworkingpos=ro.copy()
            lastworkinge=Eo
            if np.exp((Eo - En) / self.kT) > np.random.uniform():
                ro = rn.copy()
                Eo = En
        self.endT = datetime.now()
        self.log(msg='ENDING BASINHOPPING at '+self.endT.strftime('%Y-%m-%d %H:%M:%S')+':\n Number of steps to Minimum: %d\n'%self.lowstep)
        self.log(msg='Time elapsed: '+str(self.endT-self.startT)+'\n')

    def log(self, step=None, En=None, Emin=None,msg=None):
        if self.logfile is not None:
            name = self.__class__.__name__
            if step is not None:
                taim = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.logfile.write('%s: %s  step %d, energy %15.6f, emin %15.6f\n' %(name,taim, step, En, Emin))
            elif msg is not None:
                self.logfile.write(msg)
            self.logfile.flush()

    def move(self, ro):
        """Move atoms by a random step."""
        atoms = self.atoms
        # displace coordinates
        disp=np.zeros((len(ro),3))
        disp[self.adsorbate[0]:self.adsorbate[1],:3] = np.random.uniform(-1., 1., (self.adsorbate[1]-self.adsorbate[0], 3))
        rn = ro + self.dr * disp
        atoms.set_positions(rn)
        if self.cm is not None:
            cm = atoms.get_center_of_mass()
            atoms.translate(self.cm - cm)
        rn = atoms.get_positions()
        world.broadcast(rn, 0)
        atoms.set_positions(rn)
        return atoms.get_positions()

    def move_del(self, ro,vectors):
        """Displace atoms by randomly selected delocalized 'normal mode' """
        atoms=self.atoms[self.adsorbate[0]:self.adsorbate[1]]
        atms=self.atoms
        numvec=len(vectors)
        numcomb=self.numdelmodes
        while True:
            disp=np.zeros((len(ro),3))
            start=0 #find way for start to be number of stretches
            if self.constr:
                mm=atoms.get_masses()
                x0=atoms.get_positions().flatten()
                vv=VCG(atoms.get_chemical_symbols(),masses=mm)
                start=len(icSystem(vv(x0),len(atoms), masses=mm,xyz=x0).getStretchBendTorsOop()[0][0])
            if numcomb>len(range(start,numvec)):
                numcomb=len(range(start,numvec))
            w=np.random.choice(range(start,numvec),size=numcomb,replace=False)
            for i in w:
                disp[self.adsorbate[0]:(self.adsorbate[1]),:3]+=vectors[i]*np.random.uniform(-1.,1.) #this is important if there is an adsorbate.
            disp/=np.max(np.abs(disp))
            #from here on, everything is JUST COPIED from self.move(); should be sane
            rn=ro+self.dr*disp
            atms.set_positions(rn)

            if self.cm is not None:
                cm = atms.get_center_of_mass()
                atms.translate(self.cm - cm)
            rn = atms.get_positions()
            world.broadcast(rn, 0)
            atms.set_positions(rn)
            if self.check_distances(atoms):
                break
            else:
                print 'HIIIIGHWAY TO THE DANGERZONE!'
                atoms.write('Maverick.xyz')
        return atms.get_positions()

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        self.positions = positions
        self.atoms.set_positions(positions)
        ret=None
        try:
            opt = self.optimizer2(self.atoms,logfile=self.optimizer_logfile)
            opt.run(fmax=self.fmax*self.fmax_mult,steps=2000)

            opt=self.optimizer(self.atoms,logfile=self.optimizer_logfile)
            opt.run(fmax=self.fmax,steps=2000)
            if self.lm_trajectory is not None:
                self.lm_trajectory.write(self.atoms)
            self.energy = self.atoms.get_potential_energy()
            ret=self.energy
        except:
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
                # In Hotbit: "overlap matrix is not positive definite"
            ret=None
        return ret
