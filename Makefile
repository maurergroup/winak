# thctk main Makefile
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2004-2007 Christoph Scheurer
#
#   This file is part of thctk.
#
#   thctk is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   thctk is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


PYTHON = python
PLATFORM = $(shell python -c 'from distutils.util import get_platform; print get_platform()')
# FC     = ifort
FC     = efort

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
