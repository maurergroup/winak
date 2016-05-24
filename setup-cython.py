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

    package_name = 'winak'

    packages = [ package_name ]
    for package in __all__:
        packages.append(package_name + '.' + package)

    ext_modules = [
        Extension("curvilinear.cICTools", 
                    ["curvilinear/cICTools.pyx",], 
                    extra_compile_args =[], 
                    libraries = ['blas',],
                    ),
    ]

    setup(
        package_dir = { package_name: '.' },
        cmdclass = {'build_ext': build_ext},
        packages = packages,
        ext_package = package_name,
        ext_modules = ext_modules,
    )

else:
    print('Please install cython before running this file!')
