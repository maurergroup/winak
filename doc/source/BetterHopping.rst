BetterHopping
*************

BetterHopping is a variation of Basinhopping, which is a global optimization algorithm based on Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116 and David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999). It makes use of delocalized internal coordinates to reduce dissociation.

Usage
=====

All that is needed is a properly initialized ase.atoms object with a calculator::
	from ase.optimize import BFGS
	from INTERNALS.globaloptimization.betterhopping import BetterHopping
	bh = BetterHopping(atoms=molecule,
		          temperature=100 * kB,
		          dr=1.1,
		          optimizer=BFGS,
		          fmax=0.025,
		          logfile='global.log',
		          movemode=1
		          )
	bh.run(20)

Constructor Arguments
=====================

.. class:: INTERNALS.globaloptimization.BetterHopping(atoms,temperature=100*kB,optimizer=FIRE,optimizer2=FIRE,fmax=0.1,dr=0.1,logfile='-',trajectory='lowest.traj',optimizer_logfile='stdout.log',local_minima_trajectory='temp_local_minima.traj',final_minima_trajectory='final_minima.traj',adjust_cm=True,movemode=0,maxmoves=10000,dynstep=-1,numdelocmodes=1,adsorb=None)

Returns something, I guess
