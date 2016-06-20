#
# winak distutils setup
#

package_name = 'winak'

# the following provides: __version__, __revision__, __all__
execfile('__init__.py')

import ConfigParser, os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from numpy.distutils.core import setup, Extension
import numpy
numerix_macros = [('THCTK_NUMBACKEND', 1)]
numerix_include_dirs = [numpy.get_include()]

# C extensions should NOT check python types
NO_PY_CHECKS = [('NOPyChecks', None),] # use no checks (faster)

# C extensions should check python types
# NO_PY_CHECKS = [] # use checks

# The following variables define which LAPACK and BLAS libraries are to be
# used and the defaults should be overwritten in a site.cfg file.
# See site.cfg.example for an example how to do that.
config = ConfigParser.ConfigParser({
    'lapack_libs': 'lapack blas',
    'lapack_dirs': '',
    'fortran_dirs': '',
    'fortran_libs': '',
    'pardiso_dirs': '',
    'pardiso_libs': '',
    })

config.read('site.cfg')

try:
    confkey = os.environ['FC'].lower().strip()
except KeyError:
    confkey = 'DEFAULT'

try:
    tmp = config.get(confkey, 'lapack_libs')
except ConfigParser.NoSectionError:
    confkey = 'DEFAULT'

f2py_macros = []

key = 'F2PY_REPORT_ON_ARRAY_COPY'
try:
    config.get('DEFAULT', key)
    f2py_macros.append((key, None))
except:
    pass

pardiso_libs = config.get(confkey, 'pardiso_libs').split()
pardiso_dirs = config.get(confkey, 'pardiso_dirs').split()
lapack_libs = config.get(confkey, 'lapack_libs').split()
lapack_dirs = config.get(confkey, 'lapack_dirs').split()
fortran_libs = config.get(confkey, 'fortran_libs').split()
fortran_dirs = config.get(confkey, 'fortran_dirs').split()

packages = [ package_name ]
for package in __all__:
    packages.append(package_name + '.' + package)

ext_modules = [
    Extension('curvilinear.numeric.comm', ['curvilinear/numeric/comm.pyf', \
            'curvilinear/numeric/comm.f'],
        library_dirs = lapack_dirs + fortran_dirs,
        libraries = lapack_libs + fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('curvilinear.numeric.rcm', ['curvilinear/numeric/rcm.pyf', 
        'curvilinear/numeric/sparsepak_rcm.f90',],
        library_dirs = fortran_dirs,
        libraries = fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('curvilinear.numeric.icfs', ['curvilinear/numeric/icfs.pyf', \
            'curvilinear/numeric/icfs.f'],
        library_dirs = lapack_dirs + fortran_dirs,
        libraries = lapack_libs + fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('curvilinear.numeric.csrVSmsr', ['curvilinear/numeric/csrVSmsr.pyf', \
            'curvilinear/numeric/csrVSmsr.f'],
        define_macros= f2py_macros + numerix_macros),
    Extension('curvilinear.numeric.blassm', ['curvilinear/numeric/blassm.pyf', \
            'curvilinear/numeric/blassm.f'],
        define_macros= f2py_macros + numerix_macros),
#    Extension('numeric.sparslab', ['numeric/sparslab.pyf', 'numeric/sainvsr.f', 
#        'numeric/rifsr.f'],
#        include_dirs=['include'],
#        define_macros= f2py_macros + numerix_macros),
#   Extension('numeric.sainv', ['numeric/sainv.pyf', 'numeric/sainv.f'],
#   Extension('numeric.sainv', ['numeric/sainvmodule.c', 'numeric/sainv.f'],
#       include_dirs=['include'],
#       define_macros= f2py_macros + numerix_macros),
    Extension('curvilinear.numeric._numeric', ['curvilinear/numeric/thctk_numeric.c', \
            'curvilinear/numeric/colamd.c'],
        extra_compile_args =['-fopenmp'], 
        libraries = ['gomp'],
        include_dirs=['include'],
        define_macros= [('THCTK_INTERFACE', None),] + NO_PY_CHECKS  + numerix_macros),
    Extension('curvilinear._intcrd', ['curvilinear/intcrd.c'],
        include_dirs=['include'],
        undef_macros=['MAIN', 'NOPyChecks'],
        define_macros=[('THCTK_INTERFACE', None),] + numerix_macros),
    Extension("curvilinear.cICTools", ["curvilinear/cICTools.c", ]
                ),
    ]

setup(name = package_name,
      version = __version__,
      author = 'Konstantin Krautgasser, Reinhard J. Maurer',
      author_email = 'reinhard.maurer@yale.edu',
      #url = 'http://url.com/',
      package_dir = {package_name: '.'},
      description = 'python package for Global Optimization in Curvilinear Coordinates',
      long_description = 'python package for Global Optimization in Curvilinear Coordinates',
      license = 'GNU GPL',
      platforms = 'UNIX',
      packages = packages,
      ext_package = package_name,
      ext_modules = ext_modules,
    )
