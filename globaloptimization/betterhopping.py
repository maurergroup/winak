import numpy as np

from datetime import datetime
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import PickleTrajectory
from ase.vibrations import Vibrations
from ase.constraints import Hookean
#from INTERNALS.globaloptimization import Delocalizer  You don't need the delocalizer yet, right?

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
                 fmax=0.1,
                 dr=0.1,
                 logfile='-',
                 trajectory='lowest.traj',
                 optimizer_logfile='stdout.log',
                 local_minima_trajectory='temp_local_minima.traj',	#saved during the optimization in case of program crashes
                 final_minima_trajectory='final_minima.traj',	#saved at the end of the basin hopping
                 adjust_cm=True,
                 movemode=0,		#Pick a way for displacement. 0->random cartesian, 1->along vibrational modes, 2->delocalized internals
                 maxmoves=10000,	#Should prevent an issue, where you get stuck in a structure, for which get_energy() fails
                 constr=False,     	#Should a Hookean Constraint to bond length apply?
		 weighted=False,		#In case of delocalized internal displacements: do you want them to be weighted?
		 dynstep=-1,		#after what number of steps into the same minimum should the stepwidth increase? TODO
                 numdelocmodes=1,    #should a LC of modes be applied for the displacement? How many should be combined?
		 adsorb=None):	#If movemode==2, only the delocalized internals of the adsorbate are of interest. Calculate them with adsorbate.
					#adsorbate atoms have to be the first atoms in the atoms object
        Dynamics.__init__(self, atoms, logfile, trajectory)
	self.adsorbate=adsorb
        self.kT = temperature
	self.numdelmodes=numdelocmodes
        self.optimizer = optimizer
        self.fmax = fmax
        self.mm=maxmoves
        self.dr = dr
        self.lowstep=0
        self.movemode=movemode
        self.movename='Random Cartesian'
        self.constr=constr
        self.minima=[]
	self.weighted=weighted
        self.Amin=None
	self.dynst=dynstep
        if movemode==1:
            self.movename='Normal Modes'
            self.vib=Vibrations(self.atoms)
            self.vib.run()	#this seems unnecessary, but the Vibrations class doesn't initialize everything it needs
	    self.vib.read()		#in case some vibrational files exist already (from previous calculations)            	
#	elif movemode==2:
#	    self.movename='Delocalized Internals'
#	    if self.adsorbate is None:
#		    self.deloc=Delocalizer(atoms,self.weighted)
#	    else:
#              self.adsorbate.set_positions(self.atoms.get_positions()[:len(self.adsorbate)])
#              self.adsorbate.set_chemical_symbols(self.atoms.get_chemical_symbols()[:len(self.adsorbate)])
#                self.deloc=Delocalizer(self.adsorbate,self.weighted)
#	    self.vectors=self.deloc.get_vectors()
	    
	    #for i in range(len(self.vectors)):
		#self.vectors[i]/=np.max(np.abs(self.vectors[i]))	#this should ensure comparability between different movemodes. Norm wouldn't be
								#the best choice, because then the displacements get smaller for bigger molecules
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None
        self.lmfile=local_minima_trajectory
        self.optimizer_logfile = optimizer_logfile
        self.lm_trajectory = local_minima_trajectory
        if isinstance(local_minima_trajectory, str):
            self.lm_trajectory = PickleTrajectory(local_minima_trajectory,
                                                  'a', atoms)
        self.fi_trajectory=final_minima_trajectory
        self.initialize()

    def initialize(self):
        self.positions = 0.0 * self.atoms.get_positions()
        if self.constr:
            c=[]
            for x in range(len(self.atoms)):
                for y in range(x+1,len(self.atoms)):
                    d=self.atoms.get_distance(x,y)
                    if d<1.8:
                        c.append(Hookean(x,y,rt=d*1.25,k=8))     #8 is arbitrary, should depend on bond type and distance
            self.atoms.set_constraint(c)
        self.Emin = self.get_energy(self.atoms.get_positions()) or 1.e32
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.Amin=self.atoms.copy()
        if not(self.logfile is None):
            self.startT = datetime.now()
            self.logfile.write('STARTING BASINHOPPING at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+':\n Displacements: '+self.movename+' Stepsize: %.3f fmax: %.3f T: %4.2f\n'%(self.dr,self.fmax,self.kT/kB))
            self.log(-1, self.Emin, self.Emin)

    def run(self, steps):
        """Hop the basins for defined number of steps."""
        ro = self.positions
        Eo = self.get_energy(ro)
	if self.movemode==2:	
		lastworkingpos=ro.copy()
		lastworkinge=Eo
        self.minima.append(self.atoms.copy())
        rtemp=ro.copy()			#setting back to rtemp in case of get_energy() failure
        Eotemp=Eo
        Etemp=[]
        Etemp.append(Eo)        #List of minima energies, easiest way to retrieve them later without additional calculation effort
        for step in range(steps):
            En = None
            tries=0
            #self.logfile.write('Starting Step\n')
            while En is None:
                if self.movemode==0:
                    rn = self.move(ro)
                elif self.movemode==1:
                    rn=self.move_vib(ro)
		#elif self.movemode==2:
		    #rn=self.move_del(ro)
                #self.logfile.write('move done\n')
		#self.atoms.write(str(step)+'.xyz',format='xyz')
                En = self.get_energy(rn)
                tries+=1
                if tries>self.mm:
                        self.En=1e10
                        #print "fuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu"
                	ro=rtemp.copy()	#for some reason, some random steps cause get_energy() failure; seems to be a Hotbit problem
                	Eo=Eotemp      #we are going to pretend that never happened and reverse last step
                	tries=0        #I just realized setting tries to 0 is not helping at all. If it should fail again, you are DOOMED
                	if not(self.logfile is None):
                		self.logfile.write('     WARNING: last step caused get_energy() failure; Resetting step\n')
            #self.logfile.write('while loop done\n')
            self.minima.append(self.atoms.copy())
            Etemp.append(En)
            if En < self.Emin:
                # new minimum found
                self.Emin = En
                self.rmin = self.atoms.get_positions()
                self.call_observers()
                self.lowstep=step
                self.Amin=self.atoms.copy()
            self.log(step, En, self.Emin)
            accept = np.exp((Eo - En) / self.kT) > np.random.uniform()
            if accept:
                rtemp=ro.copy()
                Eotemp=Eo
                ro = rn.copy()
                Eo = En
	    if self.movemode==2:
		"""try:			
		        if self.adsorbate is None:
				self.deloc=Delocalizer(self.atoms,self.weighted)
		        else:
                                self.adsorbate.set_positions(self.atoms.get_positions()[:len(self.adsorbate)])
                                self.adsorbate.set_chemical_symbols(self.atoms.get_chemical_symbols()[:len(self.adsorbate)])
				self.deloc=Delocalizer(self.adsorbate,self.weighted)
 			self.vectors=self.deloc.get_vectors()
			lastworkingpos=self.atoms.get_positions()
			lastworkinge=Eo
			
		    	#for i in range(len(self.vectors)):
#				self.vectors[i]/=np.max(np.abs(self.vectors[i]))
		except np.linalg.linalg.LinAlgError:
			if not(self.logfile is None):
				self.logfile.write('      WARNING: Could not create delocalized coordinates. Rolling back!\n')
			ro=lastworkingpos.copy()
			Eo=lastworkinge
			self.atoms.set_positions(ro)
			if self.adsorbate is None:
				self.deloc=Delocalizer(self.atoms,self.weighted)
			else:
                                self.adsorbate.set_positions(self.atoms.get_positions()[:len(self.adsorbate)])
                                self.adsorbate.set_chemical_symbols(self.atoms.get_chemical_symbols()[:len(self.adsorbate)])
				self.deloc=Delocalizer(self.adsorbate,self.weighted)
			self.vectors=self.deloc.get_vectors()
			#for i in range(len(self.vectors)):
#				self.vectors[i]/=np.max(np.abs(self.vectors[i]))"""
        self.fi_trajectory=PickleTrajectory(self.fi_trajectory,'a')
        i=0
        for a in self.minima:
            #a.set_rmsd(a.calc_rmsd(self.Amin))
            #a.set_Edif(np.abs(Etemp[i]-self.Emin))
            self.fi_trajectory.write(a)
            i+=1
    	if not(self.logfile is None):
    		self.endT = datetime.now()
    		self.logfile.write('ENDING BASINHOPPING at '+self.endT.strftime('%Y-%m-%d %H:%M:%S')+':\n Number of steps to Minimum: %d\n'%self.lowstep)
    		self.logfile.write('Time elapsed: '+str(self.endT-self.startT)+'\n')

    def log(self, step, En, Emin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: step %d, energy %15.6f, emin %15.6f\n'
                           % (name, step, En, Emin))
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

    def move_vib(self, ro):
    	"""Move atoms by randomly selected normal modes."""
    	atoms=self.atoms	#handling the atoms that way seems wonky to say the least imho. Not gonna change it though, would mean serious work :(
    	nummodes=self.vib.getNumberOfModes()
        n=np.random.randint(1,nummodes+1) #how many modes should be considered? randint exclusive highest value
        w=np.random.choice(range(nummodes),size=n,replace=False)  #which modes should be considered?
        disp=np.zeros((len(atoms),3))
        for i in w:
            disp+=self.vib.get_mode(i)*np.random.uniform() #no arguments -> low 0.0 (inclusive) and high 1.0 (exclusive)
        maxval=np.max(np.abs(disp)) #normalize displacement; to ensure comparability with cartesian displacements in respect to stepwidth,
        disp/=maxval        #magnitude of a displacement should be smaller than 1. actual movements performed=displacement*stepwidth

        #from here on, everything is JUST COPIED from self.move(); should be sane

        #DEBUGGING
        #print 'displacing with modes '+str(w)+'\n actual disp: '+str(disp)+'\n'
        rn=ro+self.dr*disp
        atoms.set_positions(rn)
        if self.cm is not None:
            cm = atoms.get_center_of_mass()
            atoms.translate(self.cm - cm)
        rn = atoms.get_positions()
        world.broadcast(rn, 0)
        atoms.set_positions(rn)
        return atoms.get_positions()

    def move_del(self, ro):
	"""Displace atoms by randomly selected delocalized 'normal mode' """
	"""atoms=self.atoms
	numvec=len(self.vectors)
	disp=np.zeros((len(ro),3))
	numcomb=self.numdelmodes
        if self.numdelmodes>numvec:
            numcomb=numvec #for extra snafe!
        w=np.random.choice(range(numvec),size=numcomb,replace=False)
        for i in w:
            disp[:len(self.vectors[w[0]]),:3]+=self.vectors[i] #this is important if there is an adsorbate.
        maxval=np.max(np.abs(disp)) #normalize displacement; to ensure comparability with cartesian displacements in respect to stepwidth,
        #disp/=maxval        #magnitude of a displacement should be smaller than 1. actual movements performed=displacement*stepwidth


	#from here on, everything is JUST COPIED from self.move(); should be sane
	
        rn=ro+self.dr*disp
	#print disp	
        atoms.set_positions(rn)
	#print 'displacing with mode '+str(n)+'\n'
        if self.cm is not None:
            cm = atoms.get_center_of_mass()
            atoms.translate(self.cm - cm)
        rn = atoms.get_positions()
        world.broadcast(rn, 0)
        atoms.set_positions(rn)
        return atoms.get_positions()"""

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self.atoms.copy()
        atoms.set_positions(self.rmin)
        return self.Emin, atoms

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        if np.sometrue(self.positions != positions):
		self.positions = positions
		self.atoms.set_positions(positions)

		try:
			opt = self.optimizer(self.atoms,
				     logfile=self.optimizer_logfile)
			#print 'initialized'
			opt.run(fmax=self.fmax)
			#print 'run'
			if self.lm_trajectory is not None:
				self.lm_trajectory.write(self.atoms)
			self.energy = self.atoms.get_potential_energy()
			#print 'get_pot'
		except:
                        #print 'get_energy fail'
			# Something went wrong.
			# In GPAW the atoms are probably to near to each other.
			# In Hotbit: "overlap matrix is not positive definite"
			return None

        return self.energy
