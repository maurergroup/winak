Installation Instructions
*************************

1. Install the Atomic Simulation Environment (https://wiki.fysik.dtu.dk/ase/)

2. Install Python, Numpy, Scipy, Cython (use apt-get or pip in Ubuntu Linux)

3. Install SuiteSparse (Ubuntu has libsuitesparse-dev in apt-get)

4. Install scikits.sparse (adjust setup.py to point to SuiteSparse/CHOLMOD directory)

5. Test if all dependencies are fullfilled (modules can be imported in ipython session)

6. Adjust all dependency locations in setup.py and setup-cython.py of winak

7. Issue `make` in the winak parent directory and export the build/lib.linux-<version>/ directory 
   into the PYTHONPATH variable


