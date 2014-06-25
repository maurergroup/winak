from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    use_cython = True
except:
    use_cython = False
    
if use_cython:
# the following provides: __version__, __revision__, __all__
    execfile('__init__.py')

    package_name = 'INTERNALS'

    packages = [ package_name ]
    for package in __all__:
        packages.append(package_name + '.' + package)

    ext_modules = [
#        Extension("QD.offDiagonalElement", ["QD/offDiagonalElement.pyx"]),
#        Extension("QD.chebyshev", ["QD/chebyshev.pyx"]),
#        Extension("numeric.RJD", ["numeric/RJD.pyx"]),
#        Extension("numeric.cJacobiBandDiagonalization", 
#                    ["numeric/cJacobiBandDiagonalization.pyx"],
#                    extra_compile_args =[], 
#                    libraries = ['blas'],
#                    ),
#       Extension("numeric.cVarianceMinimization", 
#                   ["numeric/cVarianceMinimization.pyx"],
#                   extra_compile_args =[], 
#                   libraries = ['blas'],
#                   ),
#       Extension("numeric.cBandMatrixTools", 
#                   ["numeric/cBandMatrixTools.pyx"],
#                   extra_compile_args =[], 
#                   libraries = ['blas'],
#                   ),
#       Extension("numeric.cVariance", 
#                   ["numeric/cVariance.pyx", "numeric/cVarianceTools.c"],
#                   extra_compile_args =['-fopenmp'], 
#                   libraries = ['blas', 'gomp'],
#                   ),
#       Extension("numeric.cOffDiagLeastSquares", 
#                   ["numeric/cOffDiagLeastSquares.pyx", 
#                       "numeric/cOffDiagLeastSquaresTools.c"], 
#                   extra_compile_args =['-fopenmp'], 
#                   libraries = ['blas', 'gomp'],
#                   ),
        Extension("curvilinear.cICTools", 
                    ["curvilinear/cICTools.pyx",], 
                    extra_compile_args =[], 
                    libraries = ['blas',],
                    ),
#        Extension("numeric.cSymmetricArrayTools", 
#                    ["numeric/cSymmetricArrayTools.pyx",], 
#                    extra_compile_args =[], 
#                    ),
    ]

    setup(
        package_dir = { package_name: '.' },
        cmdclass = {'build_ext': build_ext},
        packages = packages,
        ext_package = package_name,
        ext_modules = ext_modules,
    )

