# Makefile
#
#    winak - python package for structure search and more in curvilinear coordinates
#    Copyright (C) 2016  Reinhard J. Maurer and Konstantin Krautgasser 
#    
#    This file is part of winak 
#        
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>#

PYTHON = python
PLATFORM = $(shell python -c 'from distutils.util import get_platform; print get_platform()')
#FC     = ifort
FC     = gfortran 
F77 = gfortran
F90 = gfortran

ifeq ($(FC), gfortran)
  F2PY   = --fcompiler=gnu95
endif
ifeq ($(FC), ifort)
  F2PY   = --fcompiler=intel
endif
ifeq ($(FC), efort)
  F2PY   = --fcompiler=intelem
endif
ifeq ($(FC), pgf90)
  F2PY   = --fcompiler=pg
endif
ifeq ($(FC), g95)
  F2PY   = --fcompiler=g95
endif

dirname = $(shell basename `pwd`)

default: build

build: setup.py version
	$(PYTHON) setup-cython.py build_ext
	env FC=$(FC) $(PYTHON) setup.py build build_ext $(F2PY)

sdist: setup.py version
	$(PYTHON) setup.py sdist

bdist: setup.py version build
	$(PYTHON) setup.py bdist

tar: setup.py version build
	$(PYTHON) setup.py bdist --formats tar

clean:
	-rm -rf work.pc* ifc*
	$(PYTHON) setup.py clean

build-clean: clean
	$(PYTHON) setup.py clean --all

distclean: build-clean
	rm -rf dist

version:
	@rev=`hg tip | awk '/^changeset/{print $$2}'`; \
	test -z $$rev && exit 0; \
 	sed "s%<<revision>>%$(dirname) $$rev%" < __init__.py.in > __init__.py

.PHONY: build version
