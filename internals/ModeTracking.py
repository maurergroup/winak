# thctk.QD.ModeTracking
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2006 Christoph Scheurer
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
#
#   This file is also:
#   Copyright (C) 2006 Mehdi Bounouar

"""
several classes that interface mode tracking with QC programs,
"""

from thctk.QC.Output import GAMESS_US, MolproOut
from Scientific.IO.NetCDF import NetCDFFile
from tempfile import gettempdir
import os, shutil, glob, tempfile
from thctk.numeric import *

class MOPACGradient:

    def __init__(self, atoms, x0, method = 'MNDO', bohr = 0, doCharge = 0,
        MM = False):
        import pymopac
        if method == 'MNDO':
            self.M = pymopac.MNDO(atoms, doGrad = 1, bohr = bohr, MM = MM)
        elif method == 'AM1':
            self.M = pymopac.AM1(atoms, doGrad = 1, bohr = bohr, MM = MM)
        elif method == 'PM3':
            self.M = pymopac.PM3(atoms, doGrad = 1, bohr = bohr, MM = MM)
        else:
            raise ImplementationError
        self.x0 = N.ravel(x0).astype(nxFloat)
        self.x  = N.zeros(self.x0.shape, nxFloat)

    def __call__(self, x = None):
        if x is None:
            e, g, c = self.M(self.x0)
        else:
            self.x = N.add(self.x0, x, self.x)
            e, g, c = self.M(self.x)
        return e, g, c

    def gradient(self, x = None):
        e, g, c = self(x)
        return g

class HessianMatrix:

    def __init__(self, masses, G, step = 1.0e-3, dertype = 0, grad0 = None):
        """
        G is an object which has a gradient() method
        """
        self.setMasses(masses)
        n = len(self.masses)
        self.step = step
        self.dertype = dertype
        self.shape = (n, n)
        self.G = G
        self.grad = G.gradient
        self.dx = N.zeros(n, nxFloat)
        if dertype in (0,):
            if grad0 is None:
                self.grad0 = N.zeros(n, nxFloat)
                self.grad0[:] = self.grad()
            else:
                self.grad0 = N.array(grad0).astype(nxFloat)
        else:
            self.grad0 = None

    def setMasses(self, masses):
        self.atomicMasses = N.array(masses).astype(nxFloat)
        m = N.ones((len(masses), 3), nxFloat)
        m *= self.atomicMasses[:,NewAxis]
        self.masses = N.ravel(m)
        self.masses2 = N.sqrt(self.masses)

    def matvec(self, x, y):
        norm = N.sqrt(N.dot(x, x)) # Norm
        if norm < 1e-6:
            y[:] = 0.
        else:
            u = N.divide(x, self.masses2, self.dx)
            alpha = self.step / N.sqrt(N.dot(u, u))
            print 'alpha = ', alpha
            u *= alpha
            if self.dertype == 0 : # forward
                y = N.subtract(self.grad(u), self.grad0, y)
                dx = alpha
            elif self.dertype == 1 : #  3-point central
                y = N.subtract(self.grad(u), self.grad(-u), y)
                dx = 2*alpha
            else:
                raise ImplementationError
            y /= dx
            y /= self.masses2
        return y

    def matvec_transp (self, x, y):
        return matvec(x,y)

class Amatrix:
    """
    Abstract class
    Defines a Matrix*vector product, using numerical derivatives
    Input: object (from thctk.QC.Input import GamessIn, MolproIn...)
    step: step for num. diff.
    dertype: derivative type flag
    restart: restart flag
    """
         
    def __init__ (self, Input=None, step=1.0e-3, dertype=0, restart=0):
        self.Inp = Input
        self.x0 = self.Inp.coord
        self.gradx0 = None
        self.Nq = len(self.x0)
        self.shape = (self.Nq, self.Nq) # shape attribute
        self.filename = self.Inp.filename
        self.SetMass()
        self.step = step
        self.dertype = dertype
        self.restart = restart
        self.i = 0
        self.ext = "%i" % self.i
    
    def SetMass(self):
        self.Masses = N.zeros(self.Nq, nxFloat)
        j = 0
        for k in self.Inp.masses:
            self.Masses[j:j+3] = k
            j += 3
    
    def InitNetcdf (self):
        datafile = NetCDFFile(self.filename + '.nc', 'w')
        datafile.createDimension('single', 1)
        datafile.createDimension('vector', self.Nq)
#       datafile.createDimension('dertype', self.dertype)
        datafile.createDimension('xyz', 3)
        datafile.createDimension('iter', None)
        self.datCoord = datafile.createVariable('coord', ncFloat,  ('iter', 'vector'))
#       self.coord.units = 'Bohrs'
        self.datEnergy = datafile.createVariable('Energies', ncFloat, ('iter', 'single'))
#       Energies.units = 'Hartrees'
        self.datGrad = datafile.createVariable('Gradients', ncFloat,  ('iter', 'vector'))
#       self.Dipole = datafile.createVariable('Dipole', ncFloat,  ('iter', 'xyz'))
        self.datafile = datafile
        
    def AppendToNetcdf (self):
        self.datafile = NetCDFFile(self.filename + '.nc', 'a')
        self.datCoord = self.datafile.variables['coord']
        self.datEnergy = self.datafile.variables['Energies']
        self.datGrad = self.datafile.variables['Gradients']

    def WriteToNetcdf (self, coord=None, energy=None, grad=None, dipole=None):
        self.datCoord [self.i] = coord
        self.datEnergy [self.i] = energy
        self.datGrad [self.i] = grad
#       self.Dipole [self.i] = Dipole
        self.datafile.sync()
    
    def ReadNetcdf (self, filename):
        self.readfile = NetCDFFile(filename, 'r')
        self.gradfile = self.readfile.variables['Gradients']
    
    def DoBackup(self, ext = '.bak', clobber = False):
        nc = self.filename + '.nc'
        bak = nc + ext
        if os.path.isfile(bak) and not clobber:
            raise IOError "Backup file %s already exists!" %(bak,)
        elif os.path.isfile(nc):
            oldnc = open(nc, 'r')
            fd, tmpf = tempfile.mkstemp('.tmp', 'backup_', '.')
            f = os.fdopen(fd, 'w')
            shutil.copyfileobj(oldnc, f)
            oldnc.close()
            f.close()
            os.rename(tmpf, bak)
            shutil.copystat(nc, bak)
    
    def matvec (self, x, y):
        norm = N.sqrt(N.dot(x, x)) # Norm
        if norm < 0.99 or norm > 1.01: 
            y[:] = 0.
        else:
            u = x / norm
            alpha = self.step / N.sqrt(N.dot(u, (u/self.Masses)))
            deltax =  alpha * ( u / N.sqrt(self.Masses))
            if self.restart:
                if (self.dertype == 0) and (self.i < self.Iterations -1): # forward
                    self.i += 1
                    y[:] = self.gradfile[self.i] - self.gradx0
                    y /= (alpha * N.sqrt(self.Masses))
                elif (self.dertype == 1) and (self.i < self.Iterations-1): # 3-point central
                    self.i += 2
                    y[:] = (self.gradfile[self.i-1] -  self.gradfile[self.i])
                    y /= 2 * alpha * N.sqrt(self.Masses)
                elif (self.dertype == 0):
                    self.i += 1
                    self.restart=0
                    self.readfile.close()
                    self.AppendToNetcdf()
            else:
                if self.dertype == 0 : # forward
                    y[:] = self.Function(deltax + self.x0) - self.gradx0
                    y /= (alpha * N.sqrt(self.Masses))
                elif self.dertype == 1 : #  3-point central
                    y[:] = (self.Function(deltax + self.x0) -  self.Function(-deltax + self.x0))
                    y /= 2 * alpha * N.sqrt(self.Masses)
        return y

    def matvec_transp (self, x, y):
        return matvec(x,y)

class Gamess (Amatrix):
    """ GAMESS-US specific class """
    
    def __init__ (self, Input=None, step=1.0e-3, dertype=0, restart=0):
        Amatrix.__init__ (self, Input, step, dertype, restart)
        if not self.restart:
            self.InitNetcdf()
            self.gradx0 = self.Function(self.x0)
            self.Iterations = None
        elif self.restart:
            self.DoBackup()
            if not os.path.isfile(self.filename + '.nc'):
                str = self.filename + '.nc'
                print "FILE %s NOT FOUND" %str
                raise SystemExit
            else:
                self.ReadNetcdf(self.filename + '.nc')
                self.Iterations = len(self.gradfile)
                self.gradx0 = self.gradfile[0]

    def Function (self, coord):
        ext = "%i" % self.i
        self.Inp.SetExt(ext)
        filename = self.Inp.fileIn[:-4]
        self.Inp.SetCoord (coord)
        self.Inp()
        tmpdat = gettempdir() + '/' + filename + ".dat"
        if tmpdat not in glob.glob(tmpdat):
            self.Parser = GAMESS_US(filename, datfile="")
        else:
            self.Parser = GAMESS_US(filename, datfile=tmpdat)
        self.Parser.GetGradient()
        self.Parser.GetGeometry()
        grad = N.ravel(self.Parser.grad)
        self.WriteToNetcdf(N.ravel(self.Parser.coord), self.Parser.energy, grad)
        self.i += 1
        return grad

class Molpro (Amatrix):
    
    def __init__ (self, Input=None, step=1.0e-3, dertype=0, restart=0):
        Amatrix.__init__ (self, Input, step, dertype, restart)
        
        if not self.restart:
            self.InitNetcdf()
            self.gradx0 = self.Function(self.x0)
            self.Iterations = None
        elif self.restart:
            self.DoBackup()
            if not os.path.isfile(self.filename + '.nc'):
                str = self.filename + '.nc'
                print "FILE %s NOT FOUND" %str
                raise SystemExit
            else:
                self.ReadNetcdf(self.filename + '.nc')
                self.Iterations = len(self.gradfile)
                self.gradx0 = self.gradfile[0]

    def Function (self, coord):
        ext = "%i" % self.i
        self.Inp.SetExt(ext)
        filename = self.Inp.fileIn[:-4] + ".out"
        self.Inp.SetCoord (coord)
        self.Inp()
        self.Parser = MolproOut(filename)
        self.Parser.parse()
        grad = N.ravel(self.Parser.grad[-1], nxFloat)
        coord = self.Parser.coord
        energy = self.Parser.energy[0]
        self.WriteToNetcdf(coord=N.ravel(coord), energy=energy, grad=grad)
        self.i += 1
        return grad


