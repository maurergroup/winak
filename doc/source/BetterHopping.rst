BetterHopping
*************

BetterHopping is a variation of the Basinhopping algorithm [Wales1]_, [Wales2]_. 
It makes use of delocalized internal coordinates to make global structure search more efficient.

Usage
=====

All that is needed is a properly initialized ase.atoms object with a calculator::

	from ase.optimize import BFGS
	from winak.globaloptimization.betterhopping import BetterHopping
	bh = BetterHopping(atoms=molecule,temperature=100 * kB,
		          dr=1.1,
		          optimizer=BFGS,
		          fmax=0.025,
		          logfile='global.log',
		          movemode=1
		          )
	bh.run(20)

Constructor Arguments
=====================

.. class:: winak.globaloptimization.BetterHopping(atoms,temperature=100*kB,optimizer=FIRE,optimizer2=FIRE,fmax=0.1,dr=0.1,logfile='-',trajectory='lowest.traj',optimizer_logfile='optim.log',local_minima_trajectory='temp_local_minima.traj',adjust_cm=True,movemode=0,maxmoves=10000,numdelocmodes=1,adsorb=None)

BetterHopping Object

Performs the Basinhopping algorithm for a given :class:`ase.atoms.Atoms` object.

**Parameters:**

atoms: :class:`ase.atoms.Atoms` 
	A properly initialized :class:`ase.atoms.Atoms` object with a calculator of your choice. 
temperature: float or int
	If the next displacement is performed from a new minimum (found after local optimization) or if the coordinates get set back to the previous one is determined by the following condition: ``np.exp((Eo - En) / self.kT) > np.random.uniform()``. Eo-En is the energy difference between the previous and the newly found minimum, kT is set to the temperature value given in the constructor. It should be specified in multiples of kB.
optimizer/optimizer2: :class:`ase.optimize.optimize.Optimizer` derivative
	Optimizer objects performing the local optimization. Convergence behaviour suggests the use of 2 different optimizers. One optimizer for regions far from the minimum (optimizer2) and one for regions close to the minimum (optimizer). It has proven useful to use :class:`ase.optimize.fire.FIRE` for optimizer2 and :class:`ase.optimize.BFGS` for the optimizer argument.
fmax: float
	Local optimization is performed until the maximum residual force is smaller than the value specified for fmax in eV/Angstrom. The optimizer specified in the optimizer2 argument is used until the maximum residual force is smaller than 15 times fmax.
dr: float
	This parameter is sometimes referred to as stepwidth. A displacement for the whole molecule is normed, so that the largest single change in a coordinate in x, y or z direction for any atom is at most 1 Angstrom. Afterwards the displacement is multiplied by the value specified for dr.
logfile: str
	Stdout logfile name.
trajectory: str
	Filename for trajectory file, which stores the most stable configuration.
optimizer_logfile: str
	Optimizer logfile name.
local_minima_trajectory: str
	Filename for trajectory file, which stores every minimum that was found.
adjust_cm: boolean
	Determines whether or not a center of mass adjustment should be performed after a displacement.
movemode: int
	Determines which kind of displacement will be performed. 
    * movemode=0: For every x, y and z coordinate of every atom a random number between -1.0 and +1.0 is generated and multiplied by the value specified for dr. 
    * movemode=1: Delocalized coordinates are generated for the molecule and a displacement is performed along either 1 or a linear combination of multiple coordinates. The total displacement is normed, so that the largest single change in a coordinate in x, y or z direction for any atom is at most 1 Angstrom. Afterwards the displacement is multiplied by the value specified for dr.
maxmoves: int
	In rare cases, it is not possible to displace the structure in a minimum, without the optimizer or the calculator failing. This parameter helps avoid getting stuck at a certain minimum. After the specified amount of failed displacements the structure is reset to the last minimum.
numdelocmodes: int
	If specified to a value above 1, this is the number of delocalized internal coordinates that are used in the linear combination for the displacement. Every coordinate used in the linear combination is multiplied by a random number between -1.0 and +1.0. 
adsorb: list of 2 integers
	If working with adsorbates, it is necessary to specify at what indices the adsorbate is found in the atoms object. Consider this example: The atoms object contains ammonia on a surface and the first 4 indices in the atoms object are of the adsorbed ammonia. The adsorbmask parameter would be set to (0,5).
