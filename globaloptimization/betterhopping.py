import numpy as np
from datetime import datetime
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import PickleTrajectory
from INTERNALS.curvilinear.Coordinates import DelocalizedCoordinates as DC
from INTERNALS.globaloptimization.delocalizer import *
from INTERNALS.curvilinear.Coordinates import CompleteDelocalizedCoordinates as CDC
from INTERNALS.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from INTERNALS.curvilinear.InternalCoordinates import icSystem

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
                    optimizer2=FIRE, #which optimizer to use to 15*fmax
                    fmax=0.1,
                    dr=0.1,
                    logfile='-',
                    trajectory='lowest.traj',
                    optimizer_logfile='stdout.log',
                    local_minima_trajectory='temp_local_minima.traj',	#saved during the optimization in case of program crashes
                    final_minima_trajectory='final_minima.traj',	#saved at the end of the basin hopping
                    adjust_cm=True,
                    movemode=0,		#Pick a way for displacement. 0->random cartesian, 1->delocalized internals
                    maxmoves=1000,	#Should prevent an issue, where you get stuck in a structure, for which get_energy() fails
                    dynstep=-1,		#after what number of steps into the same minimum should the stepwidth increase? TODO
                    numdelocmodes=1,    #should a LC of modes be applied for the displacement? How many should be combined?
                    adsorbmask=None):	#mask that specifies where the adsorbate is located in the atoms object (list of lowest and highest pos)
        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.adsorbate=adsorbmask
        #if adsorbmask is not None:
            #self.adsorbate=atoms[adsorbmask[0]:(adsorbmask[1]+1)]
        #else:
            #self.adsorbate=None
        self.kT = temperature
        self.numdelmodes=numdelocmodes
        self.optimizer = optimizer
        self.optimizer2=optimizer2
        self.fmax = fmax
        self.mm=maxmoves
        self.dr = dr
        self.lowstep=0
        self.movemode=movemode
        self.movename='Random Cartesian'
        self.minima=[]
        self.dynst=dynstep
        if movemode==1:
            self.movename='Delocalized Internals'
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None
        self.lmfile=local_minima_trajectory
        self.optimizer_logfile = optimizer_logfile
        self.lm_trajectory = local_minima_trajectory
        if isinstance(local_minima_trajectory, str):
            self.lm_trajectory = PickleTrajectory(local_minima_trajectory,'a', atoms)
        self.fi_trajectory=final_minima_trajectory
        self.initialize()
        #print 'initialize done'

    def check_distances(self,atoms,min=0.1):
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
        deloc=Delocalizer(atoms)
        if self.adsorbate is None:
            coords=DC(deloc.x_ref.flatten(), deloc.masses, atoms=deloc.atoms, ic=deloc.ic, u=deloc.u)
        else:
            coords=CDC(deloc.x_ref.flatten(), deloc.masses, unit=1.0, atoms=deloc.atoms, ic=deloc.ic, u=deloc.u)
        return coords.get_vectors()

    def initialize(self):
        self.startT = datetime.now()
        self.log(msg='STARTING BASINHOPPING at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+':\n Displacements: '+self.movename+' Stepsize: %.3f fmax: %.3f T: %4.2f\n'%(self.dr,self.fmax,self.kT/kB))
        self.positions = 0.0 * self.atoms.get_positions()
        self.Emin = self.get_energy(self.atoms.get_positions()) or 1.e15
        self.rmin = self.atoms.get_positions()
        self.call_observers()
        self.log(-1, self.Emin, self.Emin)

    def run(self, steps):
        """Hop the basins for defined number of steps."""
        ro = self.positions
        lastmol=ro.copy()
        Eo = self.get_energy(ro)
        lastworkingpos=ro.copy()
        lastworkinge=Eo
        self.minima.append(self.atoms.copy())
        #print 'starting for'
        for step in range(steps):
            En = None
            tries=0
            if self.movemode==1:
                atemp=self.atoms.copy()
                atemp.set_positions(ro)
                try:
                    if self.adsorbate is None:
                        vectors=self.get_vectors(atemp)
                    else:
                        vectors=self.get_vectors(atemp[self.adsorbate[0]:(self.adsorbate[1]+1)])
                except:
                    #usually the case when the molecule dissociates
                    self.log(msg='      WARNING: Could not create delocalized coordinates. Rolling back!\n')
                    self.atoms.set_positions(lastmol)
                    atemp=self.atoms.copy()
                    #atemp.set_positions(ro)
                    if self.adsorbate is None:
                        vectors=self.get_vectors(atemp)
                    else:
                        vectors=self.get_vectors(atemp[self.adsorbate[0]:(self.adsorbate[1]+1)])
                    ro=lastmol.copy()
                lastmol=ro.copy()
            #self.logfile.write('Starting Step\n')
            #print 'starting step'
            while En is None:
                if self.movemode==0:
                    rn = self.move(ro)
                elif self.movemode==1:
                    rn=self.move_del(ro,vectors)
                #self.logfile.write('move done\n')
                #print 'move done'
                self.atoms.write(str(step)+'.xyz',format='xyz')
                En = self.get_energy(rn)
                tries+=1
                if tries>self.mm:
                    ro=lastworkingpos.copy()	#for some reason, some random steps cause get_energy() failure; seems to be a Hotbit problem
                    Eo=lastworkinge      #we are going to pretend that never happened and reverse last step
                    tries=0
                    self.log(msg='     WARNING: last step caused get_energy() failure; Resetting step\n')
            #self.logfile.write('while loop done\n')
            #print 'while done'
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
            accept = np.exp((Eo - En) / self.kT) > np.random.uniform()
            if accept:
                ro = rn.copy()
                Eo = En
        self.fi_trajectory=PickleTrajectory(self.fi_trajectory,'a')
        i=0
        for a in self.minima:
            self.fi_trajectory.write(a)
            i+=1
        self.endT = datetime.now()
        self.log(msg='ENDING BASINHOPPING at '+self.endT.strftime('%Y-%m-%d %H:%M:%S')+':\n Number of steps to Minimum: %d\n'%self.lowstep)
        self.log(msg='Time elapsed: '+str(self.endT-self.startT)+'\n')

    def log(self, step=None, En=None, Emin=None,msg=None):
        if self.logfile is not None:
            name = self.__class__.__name__
            if step is not None:
                self.logfile.write('%s: step %d, energy %15.6f, emin %15.6f\n' %(name, step, En, Emin))
            elif msg is not None:
                self.logfile.write(msg)
            self.logfile.flush()

    def move(self, ro):
        """Move atoms by a random step."""
        atoms = self.atoms
        # displace coordinates
        disp = np.random.uniform(-1., 1., (len(atoms), 3))
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
        atoms=self.atoms
        numvec=len(vectors)
        numcomb=self.numdelmodes
        if self.numdelmodes>numvec:
            numcomb=numvec #for extra snafe!
        while True:
            disp=np.zeros((len(ro),3))
            w=np.random.choice(range(numvec),size=numcomb,replace=False)
            for i in w:
                disp[self.adsorbate[0]:(self.adsorbate[1]+1),:3]+=vectors[i]*np.random.uniform(-1.,1.) #this is important if there is an adsorbate.
            disp/=np.max(np.abs(disp))
            #print disp
            #from here on, everything is JUST COPIED from self.move(); should be sane
            rn=ro+self.dr*disp
            atoms.set_positions(rn)

            if self.cm is not None:
                cm = atoms.get_center_of_mass()
                atoms.translate(self.cm - cm)
            rn = atoms.get_positions()
            world.broadcast(rn, 0)
            atoms.set_positions(rn)
            if self.check_distances(atoms):
                break
            else:
                print 'HIIIIGHWAY TO THE DANGERZONE!'
                atoms.write('Maverick.xyz')
        return atoms.get_positions()

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        self.positions = positions
        self.atoms.set_positions(positions)
        ret=None
        #print 'try'
        try:
            opt = self.optimizer2(self.atoms,logfile=self.optimizer_logfile)
            #print 'initialized'
            opt.run(fmax=self.fmax*15,steps=2000)

            opt=self.optimizer(self.atoms,logfile=self.optimizer_logfile)
            opt.run(fmax=self.fmax,steps=2000)
            #print 'run'
            if self.lm_trajectory is not None:
                self.lm_trajectory.write(self.atoms)
            self.energy = self.atoms.get_potential_energy()
            ret=self.energy
            #print 'get_pot'
        except:
                #print sys.exc_info()[0]
                            #print 'get_energy fail'
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
                # In Hotbit: "overlap matrix is not positive definite"
            ret=None
        return ret
