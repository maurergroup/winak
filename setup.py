#
# thctk distutils setup
#

package_name = 'thctk'


# the following provides: __version__, __revision__, __all__
execfile('__init__.py')

import ConfigParser, os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

try:
    from scipy_distutils.core import setup, Extension
    use_numpy = False
except:
    from numpy.distutils.core import setup, Extension
    use_numpy = True

if use_numpy:
    import numpy
    numerix_macros = [('THCTK_NUMBACKEND', 1)]
    numerix_include_dirs = [numpy.get_include()]
else:
    numerix_macros = [('THCTK_NUMBACKEND', 0)]
    numerix_include_dirs = []

# C extensions should NOT check python types
NO_THCTK_PY_CHECKS = [('NOthctkPyChecks', None),] # use no checks (faster)

# C extensions should check python types
# NO_THCTK_PY_CHECKS = [] # use checks

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

packages.append(package_name + '.QD.KineticEnergy')
packages.append(package_name + '.QD.VCI')

ext_modules = [
    Extension('numeric.comm', ['numeric/comm.pyf', 'numeric/comm.f'],
        library_dirs = lapack_dirs + fortran_dirs,
        libraries = lapack_libs + fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.bspline_22', ['numeric/bspline_22.pyf',
        'numeric/bspline_22.f'],
        library_dirs = lapack_dirs + fortran_dirs,
        libraries = lapack_libs + fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.rcm', ['numeric/rcm.pyf',
        'numeric/sparsepak_rcm.f90',],
        library_dirs = fortran_dirs,
        libraries = fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.icfs', ['numeric/icfs.pyf', 'numeric/icfs.f'],
        library_dirs = lapack_dirs + fortran_dirs,
        libraries = lapack_libs + fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.ilut', ['numeric/ilut.pyf', 'numeric/ilut.f'],
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.tensorIndices', ['numeric/tensorIndices.pyf', 'numeric/tensorIndices.f'],
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.csrVSmsr', ['numeric/csrVSmsr.pyf', 'numeric/csrVSmsr.f'],
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.blassm', ['numeric/blassm.pyf', 'numeric/blassm.f'],
        define_macros= f2py_macros + numerix_macros),
    Extension('numeric.sparslab', ['numeric/sparslab.pyf', 'numeric/sainvsr.f',
        'numeric/rifsr.f'],
        include_dirs=['include'],
        define_macros= f2py_macros + numerix_macros),
#   Extension('numeric.sainv', ['numeric/sainv.pyf', 'numeric/sainv.f'],
#   Extension('numeric.sainv', ['numeric/sainvmodule.c', 'numeric/sainv.f'],
#       include_dirs=['include'],
#       define_macros= f2py_macros + numerix_macros),
    Extension('numeric._numeric', ['numeric/thctk_numeric.c', 'numeric/colamd.c'],
        extra_compile_args =['-fopenmp'],
        libraries = ['gomp'],
        include_dirs=['include'],
        define_macros= [('THCTK_INTERFACE', None),] + NO_THCTK_PY_CHECKS  + numerix_macros),
    Extension('numeric.gelsy', ['numeric/gelsy.pyf',],
        library_dirs = lapack_dirs + fortran_dirs,
        libraries = lapack_libs + fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('spectroscopy._exciton', ['spectroscopy/exciton.pyf',
        'spectroscopy/exciton.f'],
        library_dirs = fortran_dirs,
        libraries = fortran_libs,
        define_macros= f2py_macros + numerix_macros),
    Extension('QD._intcrd', ['QD/intcrd.c'],
        include_dirs=['include'],
        undef_macros=['MAIN', 'NOthctkPyChecks'],
        define_macros=[('THCTK_INTERFACE', None),] + numerix_macros),
    Extension('QD._offDiagonalElement', ['QD/offDiagonalElement.c'],
        include_dirs=['include'],
        define_macros= [] + NO_THCTK_PY_CHECKS  + numerix_macros),
    Extension('extensionTypes.symmetricArrays',
        ['extensionTypes/symmetricArrays.c',],
        include_dirs=['include'],
        define_macros= [] + NO_THCTK_PY_CHECKS  + numerix_macros),
    Extension("QD.offDiagonalElement", ["QD/offDiagonalElement.c"]),
    Extension("numeric.RJD", ["numeric/RJD.c"]),
    Extension("numeric.cJacobiBandDiagonalization",
                ["numeric/cJacobiBandDiagonalization.c"],
                ),
#   Extension("numeric.cVarianceMinimization",
#               ["numeric/cVarianceMinimization.c"],
#               ),
#   Extension("numeric.cBandMatrixTools",
#               ["numeric/cBandMatrixTools.c"],
#               ),
#   Extension("numeric.cVariance",
#               ["numeric/cVariance.c", "numeric/cVarianceTools.c"],
#               ),
#   Extension("numeric.cOffDiagLeastSquares",
#               ["numeric/cOffDiagLeastSquares.c",
#                   "numeric/cOffDiagLeastSquaresTools.c"],
#               ),
    Extension("QD.cICTools", ["QD/cICTools.c", ], libraries=['blas'],
                ),
    Extension("numeric.cSymmetricArrayTools", ["numeric/cSymmetricArrayTools.c", ]
                ),
    Extension("QD.VCI.cTools", ["QD/VCI/cTools.c",],
                extra_compile_args =['-fopenmp'],
                libraries = ['blas', 'gomp'],
                define_macros = [] + NO_THCTK_PY_CHECKS,
                ),
#   Extension("numeric.DifEqSolvers", ["numeric/DifEqSolvers.c"],
#       include_dirs=['include'],
#       undef_macros=['MAIN', 'NOthctkPyChecks'],
#       define_macros=[('THCTK_INTERFACE', None),] + numerix_macros),
    ]

if pardiso_libs: ext_modules.append(
    Extension('numeric.pardiso', ['numeric/pardiso.pyf',],
        library_dirs = pardiso_dirs + lapack_dirs + fortran_dirs,
        libraries = pardiso_libs + lapack_libs + fortran_libs,
        define_macros=[f2py_report_copy, ('long_long', '"long long"'),] + numerix_macros))

data_files = [
    (package_name, ['QC/basis.cdb',]),
    ]

setup(name = package_name,
      version = __version__,
      author = 'Christoph Scheurer',
      author_email = 'christoph.scheurer@web.de',
      url = 'http://scheurer-clark.net/thctk/',
      package_dir = { package_name: '.' },
      description = 'python package for Theoretical Chemistry',
      long_description = 'python package for Theoretical Chemistry',
      license = 'GNU GPL',
      platforms = 'POSIX',
      packages = packages,
      ext_package = package_name,
      ext_modules = ext_modules,
      data_files = data_files,
    )
