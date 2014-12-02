# Makefile
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2004-2007 Christoph Scheurer
#
#   This file was taken from Christoph Scheurers thctk package (TUM).
#

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
