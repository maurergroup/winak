# README #


### What does winak do? ###

winak is a package that contains efficient 
routines to construct and employ curvilinear 
coordinates. 

Additionally it contains routines to utilize 
such coordinates for different applications.

Currently this only amounts to their use 
in global structure optimization in chemistry.

### Dependencies ###

* Python 2.7.x
* NumPy >=1.6
* Scipy >=0.12
* Cython>=0.20
* scikits.sparse >=0.1 [ Scikits.Sparse ](https://github.com/njsmith/scikits-sparse)
  (depends itself on CHOLMOD/SuiteSparse , available via 
  pip install scikits.sparse
  )

* Atomic Simulation Environment [ ASE ](https://wiki.fysik.dtu.dk/ase/)

### Installation ###

* After installing all dependencies, just issue make 
and include build/ into PYTHONPATH

## Licensing ##

winak is licensed under the GNU General Public License, version 3 (gnu.org/licenses/gpl.html)
winak uses certain parts of the package thctk by Christoph Scheurer
and from other sources, all listed under LICENSES

### Who do I talk to? ###

r.maurer@warwick.ac.uk
konstantin.krautgasser@tum.de
