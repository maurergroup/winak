#  winak.curvilinear.InternalCoordinates
#
#  This file is part of winak.
#
#   This file was taken from Christoph Scheureres thctk package with 
#   modifications by Reinhard J. Maurer.
#
#   thctk is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
Module for handling Internal Coordinates. This module is being 
used by Coordinate classes in the Coordinates module.
"""

import string
from copy import deepcopy
import winak.curvilinear._intcrd as ic
from winak.curvilinear.cICTools import normalizeIC, denseDampIC, sparseDampIC
from winak.curvilinear.numeric._numeric import inv_L_x_cd, inv_Lt_x_cd, inv_LtL_x_cd, \
    colamd, daxpy_p
from winak.curvilinear.numeric.SparseMatrix import spmatrix, CSR, CSRd
from winak.curvilinear.numeric.icfs import dicfs, dicf
from winak.curvilinear.numeric.Matrix import regularizedInverse
from winak.curvilinear.numeric.Rotation import rigidBodySuperposition
from winak.curvilinear.numeric.MyVector import Vector
from winak.curvilinear.numeric import *
from winak.constants import covalentRadius, mass

import warnings

try:
    from MMTK.Bonds import BondList
except:
    pass
from copy import copy

from scipy.linalg import svd as SVD

class BondGenerator:
    """ 
    'BondGenerator' is a class for automatic detection of bonds in a set of 
    coordinates. 
    Create a 'BondGenerator' instance and generate bonds by calling one 
    of the two algorithms 'generateBonds' or 'generateBondsFromBoxes'. The 
    latter is preferable, because especially for large molecules it's much more
    efficient. 
    """
    def __init__(self, atoms, xyz, threshold = 0.50):
        """
        Pass a sequence of atoms and a xyz structure (in a.u.) to be used for 
        bond generation. 
        """
        if xyz.ndim == 1:
            self.xyz = xyz.reshape( (-1, 3) )
        else:
            self.xyz = xyz
        self.atoms = atoms 
        self.threshold = threshold

    def generateBonds(self):
        """
        This is a simple algorithm to find bonds between atoms defined by 
        xyz-coordinates. Implicitly the whole upper triangular part of the 
        distance matrix will be evaluated, which is quite inefficient for 
        large molecules. 
        It's recommended to us 'generateBondsFromBoxes' instead, which 
        normally is the much faster algorithm. 
        """
        xyz = self.xyz
        labels = self.atoms
        threshold = self.threshold
        bonds = set()
        n = len(xyz)
        # evaluate all distances for i < j
        for i in xrange(n-1):
            for j in xrange(i+1,n):
                # compare the distance to the sum of the covalent radii 
                # plus a additional tolerance and add a bond if 'r' is smaller 
                # than this maximal distance.
                if (N.linalg.norm(xyz[i] - xyz[j]) < 
                        ( covalentRadius[labels[i]] + 
                         covalentRadius[labels[j]] + 
                         threshold)
                        ):
                    # the bond indices are sorted because 'i' is always smaller
                    # than 'j'.
                    bonds.add( (i,j) )
        self.bonds = bonds
        return bonds

    def generateBondsFromBoxes(self, boxLength = 8.):
        """
        This is a algorithm to find bonds between atoms defined by 
        xyz-coordinates. 
        The Cartesian space will be split into a number of cubic boxes with a 
        side length defined by 'boxLength'. Next, all atoms are sorted to 
        these boxes. Now only the distances between atoms in the same box or in
        adjacent boxes will be evaluated, thereby detecting bonds.
        So by defining a 'boxLength' one should take care, that there are no 
        bonds longer than 'boxLength' because otherwise they might not be 
        detected. On the other hand it takes more time to find the bonds if the 
        boxes get larger.
        """
        self.bonds = set()
        l = float(boxLength)  # length of single box
        xyz = self.xyz
        # get minimal coordinates in x, y and z
        xyzMin = N.array([ xyz.T[i].min() for i in range(3)])
        # calculate number of boxes in all three directions (x, y and z)
        # this number is (max-min)/l + 2 because the first box has index 0 
        # and because of rounding.
        # However there is a needing for empty boxes at the boundaries, i.e
        # box[0][0][0:] = [],  box[0][0:][0] = [] ...
        # box[-1][-1][0:] = [], ...
        # so there are two more boxes in each direction, which leads to 
        # the formula (max-min)/l + 4
        nBox = [int( (xyz.T[i].max() - xyzMin[i]) / l)+4 for i in range(3) ]
        # create nBox boxes
        box = [[[ [] for k in xrange(nBox[2])] 
                     for j in xrange(nBox[1])] 
                     for i in xrange(nBox[0])]
        # sort all atoms into boxes the first boxes stay empty
        for i in xrange(len(xyz)):
            (x,y,z) = tuple( N.array(
                                ((xyz[i] - xyzMin)/l).round()+1, 
                                dtype = N.int64
                                    )
                           )            
            box[x][y][z].append(i) 
        # find all bonds inside a box and neighboured boxes therefore iterate 
        # over non empty boxes i.e index is 1 to -2.
        findBondsInBox = self._findBondsInBox
        findBondsBetweenBoxes = self._findBondsBetweenBoxes
        for x in xrange(1, nBox[0]-1): 
            for y in xrange(1, nBox[1]-1):
                for z in xrange(1, nBox[2]-1):
                    # connect all atoms inside box
                    findBondsInBox(box[x][y][z]) 
                    # and connect all atoms in _unique_ adjacent boxes, 
                    # i.e 13 neighbours which is half of the 26 neighbours 
                    # one box has in general.
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y][z])    
                    findBondsBetweenBoxes(box[x][y][z],box[x][y+1][z]) 
                    findBondsBetweenBoxes(box[x][y][z],box[x][y][z+1])
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y+1][z])
                    findBondsBetweenBoxes(box[x][y][z],box[x][y+1][z+1])
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y][z+1])
                    findBondsBetweenBoxes(box[x][y][z],box[x][y+1][z-1])
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y-1][z])
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y][z-1])
                    findBondsBetweenBoxes(box[x][y][z],box[x-1][y+1][z+1])
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y-1][z+1])
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y+1][z-1])
                    findBondsBetweenBoxes(box[x][y][z],box[x+1][y+1][z+1])
        return self.bonds

    def _findBondsInBox(self, indices):
        """
        Find all bonds between atoms in a box. Therefore evaluate the distance
        of all atoms in the box to all others.
        """
        bonds = self.bonds
        xyz = self.xyz
        labels = self.atoms
        threshold = self.threshold
        n = len(indices)
        for i in xrange(len(indices)-1):
            for j in xrange(i+1, n):
                # compare the distance to the sum of the covalent radii 
                # plus a additional tolerance and add a bond if 'r' is smaller 
                # than this maximal distance.
                p = indices[i]
                q = indices[j]
                if (N.linalg.norm(xyz[p] - xyz[q]) < 
                        ( covalentRadius[labels[p]] + 
                         covalentRadius[labels[q]] + 
                         threshold)
                        ):
					# the tuple of indices defining one bond must be sorted 
					# to have the first index smaller than the second.
                    bonds.add( (min(p,q), max(p,q)) )

    def _findBondsBetweenBoxes(self, indices1, indices2):
        """
        Find all bonds between atoms in different (adjacent) boxes. Therefore 
        evaluate distance of each atom in the first box to each atom in the 
        second box.
        """
        bonds = self.bonds
        xyz = self.xyz
        labels = self.atoms
        threshold = self.threshold
        for i in indices1: # first box
            for j in indices2: # second box
                # compare the distance to the sum of the covalent radii 
                # plus a additional tolerance and add a bond if 'r' is smaller 
                # than this maximal distance.
                if (N.linalg.norm(xyz[i] - xyz[j]) < 
                        ( covalentRadius[labels[i]] + 
                         covalentRadius[labels[j]] + 
                         threshold)
                        ):
                    # here indices must be sorted
                    bonds.add( (min(i,j), max(i,j)) )
    
    def getAdjacencyMatrix(self):
        """
        Return the adjacency matrix of the molecular graph defined by the bonds.
        """
        bonds = self.bonds
        n = len(self.atoms)
        nnz = len(bonds)
        A = spmatrix.ll_mat_sym(n, nnz)
        for ij in bonds:
            A[sorted(ij, reverse=True)] = True
        self.A = A
        Acsr = A.to_csr()
        # NOTE: the deepcopy is absolutely necessary here, because otherwise
        # the memory will be (for whatever reason) deallocated!!!
        # TODO: deepcopy shouldn't be necessary when bug in pysparse is removed
        (mx, mj, mi) = deepcopy(Acsr.matrices())
        Acsr = CSR(n = n, nnz = 2*nnz, i = mi, j = mj, x = mx, issym = 1, 
                    type = N.bool)
        self.Acsr = Acsr
        return Acsr

def dbgp(a, p=3, s=1):
    print N.array2string(a, precision=p, suppress_small=s)

def icMMTKn(obj, offset=1, dihedrals=1, impropers=0, debug=0):
    x = None
    universe = obj.universe()
    if universe is not None:
        configuration = universe.contiguousObjectConfiguration([object])
        x = N.ravel(configuration.array)
    b = BondList(obj.bonds)

    l = []
    for i in range(len(obj)): l.append([])
    ttype = 1 - offset
    for q in b:
        i = q.a1.parent.parent.sequence_number - 1
        l[i] += [ttype, q.a1.index, q.a2.index]
    ttype = 2 - offset
    for q in b.bondAngles():
        i = q.ca.parent.parent.sequence_number - 1
        l[i] += [ttype, q.a1.index, q.a2.index, q.ca.index]
    pd = 0
    id = 0
    if dihedrals:
        for q in b.dihedralAngles():
            if q.normalized:
                i = q.a1.parent.parent.sequence_number - 1
                if q.improper:
                    if impropers:
                        l[i] += [4-offset, q.a3.index, q.a2.index, q.a4.index, q.a1.index]
                        id += 1
                else:
                    l[i] += [3-offset, q.a1.index, q.a2.index, q.a3.index, q.a4.index]
                    pd += 1
    if debug:
        print "internal coordinates: %d %d %d %d\n" % (len(b),
                len(b.bondAngles()), pd, id)

    ic = []
    for k in l: ic += k
    ic.append(-offset)
    ic = N.asarray(ic, nxInt32)
    if offset: ic += N.asarray(offset, nxInt32)

    return ic, x

def icMMTK(obj, offset=1, dihedrals=1, impropers=0, debug=0):
    x = None
    universe = obj.universe()
    if universe is not None:
        configuration = universe.contiguousObjectConfiguration([object])
        x = N.ravel(configuration.array)
    b = obj.bonds

    l = []
    ttype = 1 - offset
    for q in b:
        l += [ttype, q.a1.index, q.a2.index]
    try:
        ttype = 2 - offset
        for q in b.bondAngles():
            l += [ttype, q.a1.index, q.a2.index, q.ca.index]
        pd = 0
        id = 0
        if dihedrals:
            for q in b.dihedralAngles():
                if q.normalized:
                    if q.improper:
                        if impropers:
                            l += [4-offset, q.a3.index, q.a2.index, q.a4.index, q.a1.index]
                            id += 1
                    else:
                        l += [3-offset, q.a1.index, q.a2.index, q.a3.index, q.a4.index]
                        pd += 1
    except: pass
    if debug:
        print "internal coordinates: %d %d %d %d\n" % (len(b),
                len(b.bondAngles()), pd, id)
    l.append(-offset)
    l = N.asarray(l, nxInt32)
    if offset: l += N.asarray(offset, nxInt32)

    return l, x


class ValenceCoordinateGenerator:

    def __init__(self, atoms, vdW = covalentRadius, masses = None, 
                threshold = 0.5, add_cartesians = False, manual_bonds=[]): 
        """
        Parameters:
        'threshold'       Threshold to determine bonds.
        """
        self.atoms = atoms
        self.n = len(self.atoms)
        self.vdWparameters = vdW
        self.vdW = N.zeros(self.n, nxFloat) 
        self.count = N.zeros(self.n, nxInt) 
        self.threshold = float(threshold)
        self.setvdW(vdW)
        if not add_cartesians:
            self.add_cartesians = False
        else:
            self.add_cartesians = True
        if isinstance(add_cartesians,list):
            self.cartesian_list = add_cartesians
        else:
            self.cartesian_list = range(len(self.atoms))
        self.manual_bonds = manual_bonds
        if masses is None:
            masses = N.array([mass[string.capitalize(i)] for i in atoms])
        assert len(atoms) == len(masses)
        self.masses = masses
        self.crd = None # coordinate vector

    def setvdW(self, vdW = None):
        if vdW is None:
            vdW = self.vdWparameters
        else:
            self.vdWparameters = vdW
        for i in range(self.n):
            self.vdW[i] = vdW[self.atoms[i]]

    def __call__(self, coord, bendThreshold = 170, 
                                torsionThreshold = 160, oopThreshold = 30):
        result = self.valenceBonds(coord)
        result = self.valenceAngles(bendThreshold=bendThreshold, 
                torsionThreshold=torsionThreshold, oopThreshold=oopThreshold)
        self.carts = []
        if self.add_cartesians:
            for i in self.cartesian_list:
                self.carts.append([i, 0])
            for i in self.cartesian_list:
                self.carts.append([i, 1])
            for i in self.cartesian_list:
                self.carts.append([i, 2])
        return self.toIClist()

    def valenceBonds(self, coord):
        crd = N.reshape(coord, (self.n, 3))
        # bonds = []
        cnt = self.count
        # New version using the BondGenerator
        BG = BondGenerator(atoms = self.atoms, xyz = crd, 
                                threshold = self.threshold)
        bonds = BG.generateBondsFromBoxes()
        l = []
        norm = N.linalg.norm
        for (i, j) in bonds: # i < j
            l.append( ((i, j), norm(crd[j] - crd[i])) )
            cnt[i] += 1 ; cnt[j] += 1
        bonds = l
        # Old simple version
        # for i in range(self.n):
        #     v = Vector(crd[i])
        #     ri = self.vdW[i]
        #     for j in range(i):
        #         w = Vector(crd[j])
        #         d = (v - w).length()
        #         rj = self.vdW[j]
        #         if d < (ri + rj)*self.threshold:
        #             bonds.append(((j, i), d))
        #             cnt[i] += 1
        #             cnt[j] += 1

        #add manual bonds
        for (i,j) in self.manual_bonds:
            bonds.append( ((i, j), norm(crd[j] - crd[i])) )
            cnt[i] += 1; cnt[j] +=1
        bonds.sort()
        self.bonds = bonds
        self.crd = crd
        return bonds

    def valenceAngles(self, bendThreshold = 170, torsionThreshold = 160, 
                            oopThreshold = 15):
        crd = self.crd
        bonds = self.bonds
        bends =[]
        tor_bends = []
        ba = N.cos((N.pi/180)*bendThreshold)
        ta = N.cos((N.pi/180)*torsionThreshold)
        oa = abs(N.cos( (N.pi/180) * (90-oopThreshold) ) ) 
        for (i, j), di in bonds:
            for (a, b), da in bonds:
                if i == a and j != b:
                    if j < b:
                        b1, b2, b3 = j, b, i 
                    else:
                        b1, b2, b3 = b, j, i 
                elif i == b and j != a:
                    if j < a:
                        b1, b2, b3 = j, a, i 
                    else:
                        b1, b2, b3 = a, j, i 
                elif j == a and i != b:
                    if i < b:
                        b1, b2, b3 = i, b, j 
                    else:
                        b1, b2, b3 = b, i, j 
                elif j == b and i != a:
                    if i < a:
                        b1, b2, b3 = i, a, j 
                    else:
                        b1, b2, b3 = a, i, j 
                else:
                    b1, b2, b3 = 0, 0, 0
                if b1 != b2:
                    b23 = (Vector(crd[b2])-Vector(crd[b3])).length()
                    b13 = (Vector(crd[b1])-Vector(crd[b3])).length()
                    b12 = (Vector(crd[b1])-Vector(crd[b2])).length()
                    cosg = (b13**2 + b23**2 - b12**2)/(2*b13*b23)
                    if cosg > ba:
                        bends += (b1, b2, b3),
                    if cosg > ta:
                        tor_bends += (b1, b2, b3),
        bends = list(set(bends))
        tor_bends = list(set(tor_bends))
  
        torsions = []
        for i, j, k in tor_bends:
            for a, b, c in tor_bends:
                if k == a and i == c and j != b:
                    if j < b:
                        torsions += (j, k, i, b),
                    else:
                        torsions += (b, i, k, j),
                elif k == a and j == c and i != b:
                    if i < b:
                        torsions += (i, k, j, b),
                    else:
                        torsions += (b, j, k, i),
                elif k == b and i == c and j != a:
                    if j < a:
                        torsions += (j, k, i, a),
                    else:
                        torsions += (a, i, k, j),
                elif k == b and j == c and i != a:
                    if i < a:
                        torsions += (i, k, j, a),
                    else:
                        torsions += (a, j, k, i),
        torsions = list(set(torsions))
        oops = self.getOops(oopThreshold)
        self.bends = bends
        self.tor_bends = tor_bends
        self.oops = oops
        self.torsions = torsions
        return bends, oops, torsions 
        
    def toIClist(self):
        intcrd = []
        for (i, j), d in self.bonds:
            intcrd += [1, i+1, j+1],
        for i, j, k in self.bends:
            intcrd += [2, i+1, j+1, k+1],
        for i, j, k, l in self.torsions:
            intcrd += [3, i+1, j+1, k+1, l+1],
        for i, j, k, l in self.oops:
            intcrd += [4, i+1, j+1, k+1, l+1],
        for i,j in self.carts:
            if j==0:
                intcrd += [5, i+1],
            elif j==1:
                intcrd += [6, i+1],
            else:
                intcrd += [7, i+1],
        self.LIST = copy(intcrd)
        intcrd.append([0,])
        return N.concatenate(intcrd)

    def getOops(self, oopThreshold = 15, sim = 0.2):
        """
        Parameters:
        'oopThreshold'  Threshold (in degrees) to determine, whether four atoms
                        approximately lie in a plane.
        'sim'           Factor of which masses are allowed to differ, to still 
                        count as equal.
        """
        #
        # Atoms that have exactly three bonds are possible centers for an oop:
        # count of bonds is stored in self.count...
        # Angles are calculated between atom and normal vector of 
        # plane -> 90deg - Thresh
        oa = abs(N.cos( (N.pi/180) * (90-oopThreshold) ) ) 
        count = self.count
        bonds = self.bonds
        masses = self.masses
        cands = []
        oops  = []
        for i in range(len(count)): # go over all atoms
            candi = []
            if count[i] == 3: # Has it 3 bonds?
                for (a,b), d in bonds: # To which atoms is atom i bound?
                    if i == a:
                        candi.append(b)
                    elif i  == b:
                        candi.append(a)
                candi.append(i)
            if candi != []:
                cands.append(candi)
        # Now we've got a list of possible Oops: Check if they're really in a 
        # plane        
        for i, j, k, l in cands:
            vl = Vector(self.crd[l])
            vli = (Vector(self.crd[i]) - vl).normal()
            vlj = (Vector(self.crd[j]) - vl).normal()
            vlk = (Vector(self.crd[k]) - vl).normal()
            ow = abs(vli*vlj.cross(vlk))
            # Angle between 1 atom and plane of the others < threshold?
            if ow < oa: 
                # If one atom is terminal and the others are not, we're done.
                if count[i] == 1 and count [j] != 1 and count[k] != 1:  
                    oops.append( [i, j, k,l] )
                elif count[i] != 1 and count [j] == 1 and count[k] != 1:  
                    oops.append( [j, i, k,l] )
                elif count[i] != 1 and count [j] != 1 and count[k] == 1:  
                    oops.append( [k, i, j,l] )
                # If two atoms are terminal and got similar masses, let them 
                # "wink".
                # otherwise let the lightest atom move
                elif count[i] == 1 and count [j] == 1 and count[k] != 1:  
                    if (masses[i] >= (1-sim)*masses[j] and 
                        masses[i] <= (1+sim)*masses[j]):
                        oops.append([k, i, j, l])
                    elif masses[i] < (1-sim)*masses[j]:
                        oops.append([i, j, k, l])
                    else:
                        oops.append([j, i, k, l])
                elif count[i] == 1 and count [j] != 1 and count[k] == 1:  
                    if ( masses[i] >= (1-sim)*masses[k] and 
                         masses[i] <= (1+sim)*masses[k]):
                        oops.append([j, i, k, l])
                    elif masses[i] < (1-sim)*masses[k]:
                        oops.append([i, j, k, l])
                    else:
                        oops.append([k, i, j, l])
                elif count[i] != 1 and count [j] == 1 and count[k] == 1:  
                    if ( masses[j] >= (1-sim)*masses[k] and 
                         masses[j] <= (1+sim)*masses[k] ):
                        oops.append([i, j, k, l])
                    elif masses[j] < (1-sim)*masses[k]:
                        oops.append([j, i, k, l])
                    else:
                        oops.append([k, i, j, l])
                # All atoms are terminal ->e.g. H2CO
                elif count[i] == 1 and count[j] == 1 and count[k] == 1:
                #Do two atoms have similar masses -> third moves
                    if ( masses[i] >= (1-sim)*masses[j] and  
                         masses[i] <= (1+sim)*masses[j]):
                        oops.append( [k, i, j, l] )
                    elif ( masses[i] >= (1-sim)*masses[k] and  
                           masses[i] <= (1+sim)*masses[k] ):
                        oops.append( [j, i, k, l] )
                    elif ( masses[j] >= (1-sim)*masses[k] and  
                           masses[j] <= (1+sim)*masses[k]):
                        oops.append( [i, j, k, l] )
                    # otherwise let the lightest atom move    
                    elif ( masses[i] < (1-sim)*masses[k] and 
                           masses[i] < (1-sim)*masses[j]):
                        oops.append( [i, j, k, l] )
                    elif ( masses[j] < (1-sim)*masses[i] and 
                           masses[j] < (1-sim)*masses[k] ):
                        oops.append( [j, i, k, l] )
                    elif ( masses[k] < (1-sim)*masses[i] 
                          and masses[k] < (1-sim)*masses[j] ):
                        oops.append( [k, i, j, l] )
                #else:
                    ###EXPERIMENT REINI
                    #no terminal atoms...periodic case
                    #oops.append( [i, j, k, l] )
        return oops            

# end of class ValenceCoordinateGenerator


# class PeriodicValenceCoordinateGenerator

class PeriodicValenceCoordinateGenerator(ValenceCoordinateGenerator):
    """
    constructs coordinates for periodic boundary conditions, just 
    does the same as VCG, but for a duplicated cell
    """
    def __init__(self, atoms, vdW = covalentRadius, masses = None, \
            threshold = 0.5, add_cartesians = False, 
            manual_bonds = [], cell = None,
            expand=1):

        if cell is None:
            raise ValueError('unit cell needs to be given.')
        self.cell = cell
        atoms_pbc = N.array(atoms)
        masses_pbc = N.array(masses)
        #nomenclature is x 0 y 0 z 0 , 1, -1
        if expand==2:
            self.cell_indices = N.array([
            [0,0,0],[1,0,0],[0,1,0],[0,0,1],
            [1,1,0],[1,0,1],[0,1,1],[1,1,1],
            [-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,0],
            [-1,0,-1],[0,-1,-1],[1,-1,0],[-1,1,0],
            [1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1],
            [-1,1,1],[1,-1,1],[-1,-1,1],[1,1,-1],
            [-1,1,-1],[1,-1,-1],[-1,-1,-1],
            ])
        else:
            self.cell_indices = N.array([
            [0,0,0],[1,0,0],[0,1,0],[0,0,1],
            [1,1,0],[1,0,1],[0,1,1],[1,1,1],
            ])

        self.atom_indices = N.zeros(len(atoms))
        self.manual_bonds = manual_bonds
        self.add_cartesians = add_cartesians
        for i in range(1,len(self.cell_indices)):
            atoms_pbc = N.concatenate([atoms_pbc, atoms])
            masses_pbc = N.concatenate([masses_pbc, masses])
            self.atom_indices = N.concatenate([self.atom_indices, N.ones(len(atoms))*i])
        #we duplicate the system according to periodicity
        ValenceCoordinateGenerator.__init__(self, atoms_pbc, vdW, \
                masses_pbc, threshold, add_cartesians, manual_bonds)
        
    def __call__(self,coord, **kwargs):
        """
        construct an IC List for PBC, remove duplicate DoFs
        """

        xyz = N.reshape(coord,(-1,3))
        n = len(xyz)
        
        def fold_back(ind):
            """
            folds back a set of indices by 
            """
            ii = [i/n for i in ind]
            div = min(ii)
            for j,jj in enumerate(ind):
                ind[j] -= div*n 
            return ind
        
        #duplicate coordinates
        xyz_pbc = N.empty([0,3])
        for vec in self.cell_indices:
            trans = N.dot(vec,self.cell)
            xyz_tmp = xyz + trans
            xyz_pbc = N.concatenate([xyz_pbc, xyz_tmp]) 
        self.xyz_pbc = xyz_pbc
        #construct iclist
        IClist =  ValenceCoordinateGenerator.__call__(self, xyz_pbc, **kwargs)
       
        for b, bond in reversed(list(enumerate(self.bonds))):
            indices, b_len = bond
            indices  = list(indices)
            truth = [self.atom_indices[ii] == 0 for ii in indices]
            if any(truth):
                pass
            else:
                del self.bonds[b]
        for b, bend in reversed(list(enumerate(self.bends))):
            indices = list(bend)
            truth = [self.atom_indices[ii] == 0 for ii in indices]
            if any(truth):
                pass
            else:
                del self.bends[b]
        for b, tors in reversed(list(enumerate(self.torsions))):
            indices = list(tors)
            truth = [self.atom_indices[ii] == 0 for ii in indices]
            if any(truth):
                pass
            else:
                del self.torsions[b]
        for b, oop in reversed(list(enumerate(self.oops))):
            indices = list(oop)
            truth = [self.atom_indices[ii] == 0 for ii in indices]
            if any(truth):
                pass
            else:
                del self.oops[b]
        ###ADDING CELL DoFs
        exists1=False
        exists2=False
        exists3=False
        for b, bond in enumerate(self.bonds):
            i, j = bond[0]
            if i==0 and j==n:
                exists1 = True
            if i==0 and j==2*n:
                exists2 = True
            if i==0 and j==3*n:
                exists3 = True
        if not exists1:
            self.bonds.append([(0, n),N.linalg.norm(self.cell[0])])    
        if not exists2:
            self.bonds.append([(0, 2*n),N.linalg.norm(self.cell[1])])    
        if not exists3:
            self.bonds.append([(0, 3*n),N.linalg.norm(self.cell[2])])    
        #exists1=False
        #exists2=False
        #exists3=False
        #for b, bend in enumerate(self.bends):
            #i, j, k = bend
            #if i==n and j==2*n and k==0:
                #exists1=True
            #if i==n and j==3*n and k==0:
                #exists2=True
            #if i==2*n and j==3*n and k==0:
                #exists3=True
        #if not exists1:
            #self.bends.append([n, 2*n, 0])    
        #if not exists2:
            #self.bends.append([n, 3*n, 0])    
        #if not exists3:
            #self.bends.append([2*n, 3*n, 0])    
        
        ####delete bonds which do not reach into unit cell
        ##ADDING CELL BONDS
        #bonds = []
        ##eliminate duplicate ones
        #for b,bond in enumerate(self.bonds):
            #indices, b_len = bond
            #indices = list(indices)
            ##go through all other bonds and see if there are duplicates
            #take_it = True
            #for bb, bbond in enumerate(bonds):
                #tmp_ind, bb_len = bbond
                #tmp_ind = list(tmp_ind)
                #if tmp_ind == indices:
                    #take_it = False 
                #elif tmp_ind==fold_back(indices):
                    #take_it = False
                #else:
                    #pass
            #if take_it:
                #bonds.append([tuple(fold_back(indices)),b_len])
        #self.bonds = bonds

        ####delete bends which do not reach into unit cell
        #bends = []
        #for b, bend in enumerate(self.bends):
            #indices = list(bend)
            #take_it = True
            #for bb, bbend in enumerate(bends):
                #tmp_ind = bbend
                #if tmp_ind == indices:
                    #take_it = False 
                #elif tmp_ind==fold_back(indices):
                    #take_it = False
                #else:
                    #pass
            #if take_it:
                #bends.append(fold_back(indices))
        #self.bends = bends

        ####delete torsions which do not reach into unit cell
        #torsions = []
        #for t, torsion in enumerate(self.torsions):
            #indices = list(torsion)
            #take_it = True
            #for tt, ttors in enumerate(torsions):
                #tmp_ind = ttors 
                #if tmp_ind == indices:
                    #take_it = False 
                #elif tmp_ind==fold_back(indices):
                    #take_it = False
                #else:
                    #pass
            #if take_it:
                #torsions.append(fold_back(indices))
        #self.torsions = torsions
        ####delete oop angles which do not reach into unit cell
        #oops = []
        #for o, oop in enumerate(self.oops):
            #indices = list(oop)
            #take_it = True
            #for oo, ooop in enumerate(oops):
                #tmp_ind = ooop
                #if tmp_ind == indices:
                    #take_it = False 
                #elif tmp_ind==fold_back(indices):
                    #take_it = False
                #else:
                    #pass
            #if take_it:
                #oops.append(fold_back(indices))
        #self.oops = oops
        #for c, cart in reversed(list(enumerate(self.carts))):
            #indices = list(cart)
            #if self.atom_indices[indices[0]] == 0:
                #pass
            #else:
                #del self.carts[c]
       
        return self.toIClist()

class PeriodicValenceCoordinateGenerator2(PeriodicValenceCoordinateGenerator):
    def __call__(self,coord, **kwargs):
        """
        construct an IC List for PBC, remove duplicate DoFs
        """

        xyz = N.reshape(coord,(-1,3))
        n = len(xyz)
        
        def fold_back(ind):
            """
            folds back a set of indices by 
            """
            ii = [i/n for i in ind]
            div = min(ii)
            for j,jj in enumerate(ind):
                ind[j] -= div*n 
            return ind
        
        #duplicate coordinates
        xyz_pbc = N.empty([0,3])
        for vec in self.cell_indices:
            trans = N.dot(vec,self.cell)
            xyz_tmp = xyz + trans
            xyz_pbc = N.concatenate([xyz_pbc, xyz_tmp]) 
        self.xyz_pbc = xyz_pbc
        #construct iclist
        IClist =  ValenceCoordinateGenerator.__call__(self, xyz_pbc, **kwargs)
       
        #for b, bond in reversed(list(enumerate(self.bonds))):
            #indices, b_len = bond
            #indices  = list(indices)
            #truth = [self.atom_indices[ii] == 0 for ii in indices]
            #if any(truth):
                #pass
            #else:
                #del self.bonds[b]
        #for b, bend in reversed(list(enumerate(self.bends))):
            #indices = list(bend)
            #truth = [self.atom_indices[ii] == 0 for ii in indices]
            #if any(truth):
                #pass
            #else:
                #del self.bends[b]
        #for b, tors in reversed(list(enumerate(self.torsions))):
            #indices = list(tors)
            #truth = [self.atom_indices[ii] == 0 for ii in indices]
            #if any(truth):
                #pass
            #else:
                #del self.torsions[b]
        #for b, oop in reversed(list(enumerate(self.oops))):
            #indices = list(oop)
            #truth = [self.atom_indices[ii] == 0 for ii in indices]
            #if any(truth):
                #pass
            #else:
                #del self.oops[b]
        ###ADDING CELL DoFs
        #exists1=False
        #exists2=False
        #exists3=False
        #for b, bond in enumerate(self.bonds):
            #i, j = bond[0]
            #if i==0 and j==n:
                #exists1 = True
            #if i==0 and j==2*n:
                #exists2 = True
            #if i==0 and j==3*n:
                #exists3 = True
        #if not exists1:
            #self.bonds.append([(0, n),N.linalg.norm(self.cell[0])])    
        #if not exists2:
            #self.bonds.append([(0, 2*n),N.linalg.norm(self.cell[1])])    
        #if not exists3:
            #self.bonds.append([(0, 3*n),N.linalg.norm(self.cell[2])])    
        #exists1=False
        #exists2=False
        #exists3=False
        #for b, bend in enumerate(self.bends):
            #i, j, k = bend
            #if i==n and j==2*n and k==0:
                #exists1=True
            #if i==n and j==3*n and k==0:
                #exists2=True
            #if i==2*n and j==3*n and k==0:
                #exists3=True
        #if not exists1:
            #self.bends.append([n, 2*n, 0])    
        #if not exists2:
            #self.bends.append([n, 3*n, 0])    
        #if not exists3:
            #self.bends.append([2*n, 3*n, 0])    
        
        ####delete bonds which do not reach into unit cell
        ##ADDING CELL BONDS
        bonds = []
        #eliminate duplicate ones
        for b,bond in enumerate(self.bonds):
            indices, b_len = bond
            indices = list(indices)
            #go through all other bonds and see if there are duplicates
            take_it = True
            for bb, bbond in enumerate(bonds):
                tmp_ind, bb_len = bbond
                tmp_ind = list(tmp_ind)
                if tmp_ind == indices:
                    take_it = False 
                elif tmp_ind==fold_back(indices):
                    take_it = False
                else:
                    pass
            if take_it:
                bonds.append([tuple(fold_back(indices)),b_len])
        self.bonds = bonds

        ###delete bends which do not reach into unit cell
        bends = []
        for b, bend in enumerate(self.bends):
            indices = list(bend)
            take_it = True
            for bb, bbend in enumerate(bends):
                tmp_ind = bbend
                if tmp_ind == indices:
                    take_it = False 
                elif tmp_ind==fold_back(indices):
                    take_it = False
                else:
                    pass
            if take_it:
                bends.append(fold_back(indices))
        self.bends = bends

        ###delete torsions which do not reach into unit cell
        torsions = []
        for t, torsion in enumerate(self.torsions):
            indices = list(torsion)
            take_it = True
            for tt, ttors in enumerate(torsions):
                tmp_ind = ttors 
                if tmp_ind == indices:
                    take_it = False 
                elif tmp_ind==fold_back(indices):
                    take_it = False
                else:
                    pass
            if take_it:
                torsions.append(fold_back(indices))
        self.torsions = torsions
        ###delete oop angles which do not reach into unit cell
        oops = []
        for o, oop in enumerate(self.oops):
            indices = list(oop)
            take_it = True
            for oo, ooop in enumerate(oops):
                tmp_ind = ooop
                if tmp_ind == indices:
                    take_it = False 
                elif tmp_ind==fold_back(indices):
                    take_it = False
                else:
                    pass
            if take_it:
                oops.append(fold_back(indices))
        self.oops = oops
        
        return self.toIClist()


class icSystem:

    def __init__(self, intcrd, natom, xyz = None, masses = None):
        if intcrd[-1] > 0:
            intcrd = list(intcrd)
            intcrd.append(0)
        self.ic = N.array(intcrd).astype(nxInt32)
        self.atoms_in_type = (2, 3, 4, 4, 1, 1, 1)
        self.natom = natom
        #get Bmatrix dimensions and number of DoFs Bmatrix has n times Bnnz
        self.Bnnz, self.n = ic.Bmatrix_nnz(self.ic)
        self.nx = 0
        if xyz is not None:
            self.xyz = N.ravel(xyz).astype(nxFloat)
            self.nx = len(self.xyz)
            if self.nx != 3*self.natom:
                raise IndexError("Array dimension of xyz does not match natom")
        else: self.xyz = xyz
        if masses is not None:
            if len(masses) != self.natom:
                raise IndexError("Array dimension of masses does not match natom")
            self.masses = N.array(masses).astype(nxFloat)
        else: self.masses = masses
        self.typeList = self.types()
        if len(self.typeList[2]) > 0:
            # function 'dphi_mod_2pi' of C module '_intcrd' needs an int32 array
            self.torsions = N.array(self.typeList[2], dtype = N.int32)[:,0]
        else:
            self.torsions = None
        self.initA()

    def __str__(self, lineBreak = True):
        s = []
        i = 0
        ic = self.ic
        ind = 0
        internals = self()
        while i < len(ic):
            if ic[i] == 0:
                break
            elif ic[i] == 1:
                s.append('%3i str.(%3i, %3i) = % 6.5f' 
                    %(ind, ic[i+1], ic[i+2], internals[ind]) )
                i += 3
            elif ic[i] == 2:
                s.append('%3i bend(%3i, %3i, %3i) = % 6.5f' 
                    %(ind, ic[i+1], ic[i+2], ic[i+3], internals[ind]) )
                i += 4
            elif ic[i] == 3:
                s.append('%3i tors.(%3i, %3i, %3i, %3i) = % 6.5f' 
                    %(ind, ic[i+1], ic[i+2], ic[i+3], ic[i+4], internals[ind]) )
                i += 5
            elif ic[i] == 4:
                s.append('%3i oop.(%3i, %3i, %3i, %3i) = % 6.5f' 
                    %(ind, ic[i+1], ic[i+2], ic[i+3], ic[i+4], internals[ind]) )
                i += 5
            elif ic[i] == 5:
                s.append('%1i cart x(%3i) = % 6.5f' 
                    %(ind, ic[i+1], internals[ind]) )
                i += 2
            elif ic[i] == 6:
                s.append('%3i cart y(%3i) = % 6.5f' 
                    %(ind, ic[i+1], internals[ind]) )
                i += 2
            elif ic[i] == 7:
                s.append('%3i cart z(%3i) = % 6.5f' 
                    %(ind, ic[i+1], internals[ind]) )
                i += 2
            else:
                break
            ind += 1
        if lineBreak:
            return 'icSystem(\n' +  '\n'.join(s) + '\n)'
        else:
            return 'icSystem(<' +  '>, <'.join(s) + '>)'

    def __len__(self):
        return self.n

    def __call__(self, x = None):
        if x is None: x = self.xyz
        if x is None: return None
        if not hasattr(self, 'crd'): self.crd = N.zeros(self.n, nxFloat)
        ic.internals(x, self.ic, self.crd)
        return self.crd 

    def getStretchBendTorsOop(self):
        """
        Return a tuple, where each element is a list containing the indices
        of the internal coordinates stretch, bend, torsion and oop. 
        """
        stretch = []; bend = []; tors = []; oop = []; atoms=[]
        icList = []; 
        i = 0
        ic = self.ic
        index = 0
        while i < len(ic):
            if ic[i] == 0: break
            elif ic[i] == 1: 
                stretch.append(index) 
                icList.append(ic[i+1:i+3])
                i += 3
            elif ic[i] == 2: 
                bend.append(index)    
                icList.append(ic[i+1:i+4])
                i += 4
            elif ic[i] == 3: 
                tors.append(index)    
                icList.append(ic[i+1:i+5])
                i += 5
            elif ic[i] == 4: 
                oop.append(index)     
                icList.append(ic[i+1:i+5])
                i += 5
            elif ic[i] == 5 or ic[i] == 6 or ic[i] ==7:
                atoms.append(index)
                icList.append(ic[i+1:i+2])
                i += 2
            else: break
            index += 1
        stretchBendTorsOop = (
            N.array(stretch, dtype = N.int32), 
            N.array(bend, dtype = N.int32), 
            N.array(tors, dtype = N.int32), 
            N.array(oop, dtype = N.int32),
            )
        return (stretchBendTorsOop, icList)
    
    def getStretchBendTorsOopCart(self):
        """
        Return a tuple, where each element is a list containing the indices
        of the internal coordinates stretch, bend, torsion and oop. 
        """
        stretch = []; bend = []; tors = []; oop = []; atoms=[]
        icList = []; 
        i = 0
        ic = self.ic
        index = 0
        while i < len(ic):
            if ic[i] == 0: break
            elif ic[i] == 1: 
                stretch.append(index) 
                icList.append(ic[i+1:i+3])
                i += 3
            elif ic[i] == 2: 
                bend.append(index)    
                icList.append(ic[i+1:i+4])
                i += 4
            elif ic[i] == 3: 
                tors.append(index)    
                icList.append(ic[i+1:i+5])
                i += 5
            elif ic[i] == 4: 
                oop.append(index)     
                icList.append(ic[i+1:i+5])
                i += 5
            elif ic[i] == 5 or ic[i] == 6 or ic[i] == 7:
                atoms.append(index)
                icList.append(ic[i+1:i+2])
                i += 2
            else: break
            index += 1
        stretchBendTorsOop = (
            N.array(stretch, dtype = N.int32), 
            N.array(bend, dtype = N.int32), 
            N.array(tors, dtype = N.int32), 
            N.array(oop, dtype = N.int32),
            N.array(atoms, dtype = N.int32)
            )
        return (stretchBendTorsOop, icList)

    def getBendsInOthers(self):
        """
        Return a dictionary containing the indices of bends as keys and 
        sets of other coordinates, that contain these bends as subunits.
        """
        ic = self.ic
        ((stretch, bend, tors, oop), icList) = self.getStretchBendTorsOop()
        bendsInOthers = {}
        for i in bend:
            b = icList[i]
            b = (min(b[0], b[1]), max(b[0], b[1]), b[2])
            for j in tors:
                t = icList[j]
                t1 = (min(t[0], t[2]), max(t[0], t[2]), t[1])
                t2 = (min(t[1], t[3]), max(t[1], t[3]), t[2])
                if (t1 == b) or (t2 == b): 
                    if i in bendsInOthers:
                        bendsInOthers[i].add(j)
                    else:
                        bendsInOthers[i] = set([j])
            for j in oop:
                o = icList[j]
                o = (min(o[1], o[2]), max(o[1], o[2]), o[3])
                if o == b: 
                    if i in bendsInOthers:
                        bendsInOthers[i].add(j)
                    else:
                        bendsInOthers[i] = set([j])
        return bendsInOthers

    def types(self):
        """
        Return a list, where each element is a list containing all internal 
        coordinates of one type. 
        (Same as 'getStretchBendTorsOop' but working for all internal 
        coordinates defined in 'atoms_in_type'.)
        """
        t = map(lambda x: x+1, self.atoms_in_type)
        l = []
        for k in range(len(t)): l.append([])
        c = self.ic
        i = 0
        j = 0
        while 1:
            k = c[i] - 1
            if k < 0: break
            l[k].append((j,i))
            j += 1
            i += t[k]
        return l

    def evalB(self, sort = 0):
        if not hasattr(self, 'B'):
            self.B = CSR(n=self.n, m=self.nx, nnz=self.Bnnz, type = nxFloat)
        ic.Bmatrix(self.xyz, self.ic, self.B.x, self.B.j, self.B.i, sort)
       
        return self.B

    def evalBt(self, update = 0, perm = 0):
        if not hasattr(self, 'Bt'):
            self.Bt = self.B.transpose()
            #self.Bt = CSR(n=self.nx, m=self.n, nnz=self.Bnnz, type = nxFloat)
        #if not hasattr(self, 'colperm') or not perm:
            #ic.Btrans(self.Bnnz, self.n, self.nx, update, self.B.x, self.B.j,
                #self.B.i, self.Bt.x, self.Bt.j, self.Bt.i)
        #else:
            #ic.Btrans_p(self.Bnnz, self.n, self.nx, update, self.colperm,
                #self.B.x, self.B.j, self.B.i, self.Bt.x, self.Bt.j, self.Bt.i)
        return self.Bt

    def colamd(self, inverse = 0):
        # use Bt since colamd expects CSC format
        self.colperm = colamd(self.n, self.Bt.j, self.Bt.i)[:-1]
        if inverse: self.inverseP()

    def connectivity(self):
        self.cj, self.ci = ic.symbolicAc(self.ic, self.natom)

    def evalA(self, diag = 1, sort = 0 , force = 0):
        if not hasattr(self, 'A') or force == 1:
            from winak.curvilinear.numeric.SparseMatrix import AmuB
            self.A = AmuB(self.evalBt(perm=0),self.B)
            #if not hasattr(self, 'colperm'):
                #nnz = self.nx + 9*(self.ci[-1]/2)
                #if (diag == 0): nnz += self.nx
                #aj = N.zeros(nnz, nxInt)
                #ai = N.zeros(self.nx+1, nxInt)
                #ic.conn2crd(self.natom, diag, self.cj, self.ci, aj, ai)
            #else:
                #aj, ai = ic.conn2crd_p(self.natom, diag, self.cj, self.ci,
                                       #self.colperm, sort)
            #self.Annz = ai[-1]
            #if diag:
                #self.A = CSRd(n=self.nx, nnz=self.Annz, i=ai, j=aj, type=nxFloat)
            #else:
                #self.A = CSR(n=self.nx, m=self.nx, nnz=self.Annz, i=ai, j=aj, type=nxFloat)
        #if hasattr(self.A, 'd'):
            #ic.BxBt_d(self.nx, self.Bt.x, self.Bt.j, self.Bt.i, self.Annz,
                      #self.A.x, self.A.j, self.A.i, self.A.d)
        #else:
            #ic.BxBt(self.nx, self.Bt.x, self.Bt.j, self.Bt.i,
                    #self.A.x, self.A.j, self.A.i, self.Annz, 0)
        return self.A
    
    def evalAt(self, diag = 1, sort = 0 , force = 0):
        if not hasattr(self, 'At') or force == 1:
            from winak.curvilinear.numeric.SparseMatrix import AmuB
            self.At = AmuB(self.B, self.evalBt(perm=0))
            #if not hasattr(self, 'colperm'):
                #nnz = self.nx + 9*(self.ci[-1]/2)
                #if (diag == 0): nnz += self.nx
                #aj = N.zeros(nnz, nxInt)
                #ai = N.zeros(self.nx+1, nxInt)
                #ic.conn2crd(self.natom, diag, self.cj, self.ci, aj, ai)
            #else:
                #aj, ai = ic.conn2crd_p(self.natom, diag, self.cj, self.ci,
                                       #self.colperm, sort)
            #self.Annz = ai[-1]
            #if diag:
                #self.A = CSRd(n=self.nx, nnz=self.Annz, i=ai, j=aj, type=nxFloat)
            #else:
                #self.A = CSR(n=self.nx, m=self.nx, nnz=self.Annz, i=ai, j=aj, type=nxFloat)
        #if hasattr(self.A, 'd'):
            #ic.BxBt_d(self.nx, self.Bt.x, self.Bt.j, self.Bt.i, self.Annz,
                      #self.A.x, self.A.j, self.A.i, self.A.d)
        #else:
            #ic.BxBt(self.nx, self.Bt.x, self.Bt.j, self.Bt.i,
                    #self.A.x, self.A.j, self.A.i, self.Annz, 0)
        return self.At

    def cholesky(self, A=None, p = 10, eps = 1.0e-6, fout = 1, lambd = 0.0001):
        if A is None:
            A = self.A
        else:
            A = A
        from scikits.sparse.cholmod import cholesky
        from winak.curvilinear.numeric.csrVSmsr import csrcsc
        from scipy.sparse import csc_matrix
        bx, bj, bi = csrcsc(A.n, A.m, A.x, A.j+1, A.i+1)
        bj -= 1
        bi -= 1
        AA = csc_matrix((bx, bj, bi),shape=(A.n, A.m))
        try:
            self.factor = cholesky(AA,lambd, mode='supernodal')
        except:
            raise RuntimeError('A is not positive definite. Increase lambd')
        self.L = self.factor
        #Lx, Lj, Li, Lnnz, (n, m) = L.data, L.indices, L.indptr, L.nnz, L.shape
        #nnz = Lnnz
        #if not hasattr(self, 'L') or len(self.L.x) < nnz:
            #self.L = CSR(n=n, m=m, nnz=Lnnz, i=Li, j=Lj, type=nxFloat)
            #self.L.x = Lx
        return 1 

    def inverseL(self, r):
        L = self.L
        return self.L.solve_L(r)

    def inverseLt(self, r):
        L = self.L
        return self.L.solve_Lt(r)

    def inverseA(self, r):
        L = self.L
        return self.L.solve_A(r)

    def precon_transp(self, x, y = None):
        if y is None:
            self.inverseA(x)
        else:
            y[:] = x
            self.inverseA(y)

    def precon(self, x, y = None):
        if y is None:
            self.inverseA(x)
        else:
            y[:] = x
            self.inverseA(y)

    def inverseP(self):
        p = self.colperm
        self.ip = p
        n = len(p)
        ip = N.zeros(n, nxInt)
        for i in range(n): ip[p[i]] = i
        self.colperm = ip

    def initA(self):
        self.connectivity()
        self.evalB()
        self.evalBt(perm = 0)
        #self.colamd(inverse=1)
        #self.evalBt()

    @staticmethod
    def _printWarning(i, b_i, warnType):
        if warnType == '0':
            warnings.warn("\n\tBend (int. coord. %i) " %i + 
             "is close to zero (value is: %3.2f).\n\t" %b_i + 
             "(Damping this coordinate and coordinates " + 
             "this one beeing part of.)", 
              Warning, 3)
        elif warnType == 'pi':
            warnings.warn("\n\tBend (int. coord. %i) " %i + 
             "is close to pi (value is: %3.2f).\n\t" %b_i + 
             "(Damping this coordinate and coordinates " + 
             "this one beeing part of.)", 
              Warning, 3)
        elif warnType == 'pi/2':
            warnings.warn("\n\tOut of plain (int. coord. %i) " %i + 
             "is close to (+-)pi/2 (value is: %3.2f).\n\t" %b_i + 
             "(Damping this coordinate.)", 
              Warning, 3)


    def biInit(self):
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx, nxFloat)
        if not hasattr(self, 'tmp_icArrays'):
            # here we create some arrays to be used during the back iteration
            # by the c methods. 
            # Therefore the indices of stretches, bends ... are stored. 
            # the format for the bends is special: 
            # the array 'cBends' starts with a 0, then the index of the first 
            # bend follows. The next value (minus the previous value) in the 
            # array gives the number of coordinates, this bend is a subunit 
            # of (i.e. some torsions or oops). 
            # After that the index of the next bend follows. 
            # The indices of the specified coordinates, this bend is a subunit
            # of, are stored in 'cBendsInOthers'. 
            # An example for bends with indices 5, 8 would be
            # cBends = [0, 5, 2, 8, 3]
            # cBendsInOthers = [ 3, 4, 1]
            # where in this example, the bend 5 is a subunit of coordinates 
            # 3 and 4, and bend 8 is a subunit of coordinate 1. 
            ((stretch, bends, tors, oop), tmp) = self.getStretchBendTorsOop()
            bendsInOthers = self.getBendsInOthers()
            cBends = N.empty(2*len(bends) + 1, dtype = N.int)
            cBends[0] = 0.
            cBendsInOthers = []
            j = 0
            for i in range(len(bends)):
                cBends[2*i+1] = bends[i]
                try: 
                    b = bendsInOthers[bends[i]]
                    j += len(b)
                    cBendsInOthers.extend(b)
                except KeyError: 
                    pass
                cBends[2*i+2] = j

            self.tmp_icArrays = (
                N.array(stretch, dtype = N.int), 
                N.array(cBends, dtype = N.int), 
                N.array(cBendsInOthers, dtype = N.int), 
                N.array(tors, dtype = N.int), 
                N.array(oop, dtype = N.int), 
                )

    def denseBackIteration(self, q, maxiter = 200, eps = 1e-6, initialize = 1,
            inveps = 1e-9, maxStep = 0.5, warn = True, 
            RIIS = True, RIIS_start = 20, RIIS_maxLength = 6, RIIS_dim = 4, 
            restart = True, dampingThreshold = 0.1, maxEps = 1e-6, 
            col_constraints=[], row_constraints=[]):
        """
        Find Cartesian coordinates belonging to the set of internal coordinates
        'q' in an iterative way. 
        Parameters are:
        'inveps'         Parameter for the regularized inverse.
        'maxStep'        Maximal step during iteration. (Don't make 'maxStep'
                         too small, when dealing with convergence problems, 
                         try to increase the 'RIIS_dim' instead, since
                         RIIS will only work properly if steps are not scaled 
                         down. But also don't make maxStep too small, 
                         since then you might converge on spurious solutions.
        'RIIS'           Perform Regularized Inversion of the Iterative Subspace
                         to accelerate convergence.
        'RIIS_start'     Start RIIS after given number of steps.
        'RIIS_maxLength' maximal length of spanning set for the iterative 
                         subspace.
        'RIIS_dim'       Parameter for the Tikhonov regularization, so 
                         indirectly the dimension of space that will be 
                         inverted is determined here.
        """
        assert isinstance(maxStep, float)
        assert maxStep > eps
        self.biInit()
        icArrays = self.tmp_icArrays

        if warn: printWarning = self._printWarning
        else:    printWarning = lambda *x: None

        # self.q = q # DEBUGGING
        # self.dx = [] # DEBUGGING
        # bendsInOthers = self.getBendsInOthers() # DEBUGGING
        # ((stretch, bend, tors, oop), tmp) = self.getStretchBendTorsOop() # DEBUGGING

        dq = self.tmp_q
        dx = self.tmp_x
        xn = self.xyz
        if initialize: self.initA()

        n = 0
        nScalings = 0 
        neps = eps*eps*self.nx
        xList = [] # geoms for RIIS
        dxList = [] # error vectors for RIIS
        RIIS_on = False

        normalizeIC(q, *icArrays)

        if self.torsions is not None: torsions = N.asarray(self.torsions, 
                                                            dtype = N.int32)
        else:                         torsions = None

        while 1:
            n += 1
            qn = self()

            # self.qn = qn # DEBUGGING

            # determine new dq by RIIS or traditional 
            # RIIS will be performed, if norm of last error vector is 
            # larger than the one of the second last vector and a sufficient 
            # number of vectors is available to do an extrapolation
            dq = N.subtract(q, qn, dq)
            if RIIS: 
                if not RIIS_on and n >= RIIS_start:
                    RIIS_on = True
                    # print("Turning on RIIS in step %i" %n)

            # check phase 
            if torsions is not None: dq = ic.dphi_mod_2pi(dq, torsions)

            # step width control
            dqMax = N.max(N.abs(dq))
            if dqMax > maxStep:
                # print("scaling step %i" %n)
                nScalings += 1
                dq *= maxStep/dqMax

            B = self.evalB()
            B = B.full()
            
            
            # self.Bfull = B # DEBUGGING

            # check all coordinates, if they are close to a singularity
            # i.e. stretches -> 0. , bends -> pi, bends of tors -> 0, pi
            # out of plains -> -pi/2, pi/2 or bends of out of plains -> 0., pi
            # If they get close, damp them away with 
            denseDampIC(printWarning, qn, dq, B, dampingThreshold, *icArrays)

            Bt = self.evalBt(update=1,perm=0).full()
            A = N.dot(Bt, B)
            self.Ainv = regularizedInverse(A, eps = inveps)

            for c in row_constraints:
                dq[c] = 0.0

            dx[:] = N.dot(self.Ainv, N.dot(Bt, dq))
            for c in col_constraints:
                dx[c] = 0.0
            # self.dx.append(dx.copy()) # for DEBUGGING

            if RIIS and n >= RIIS_start - RIIS_maxLength:
                dxList.append(dx.copy().reshape((-1,3))) # store dx for RIIS
                if len(xList) > 0:
                    xList.append(
                        rigidBodySuperposition(
                            xn.reshape((-1,3)), xList[-1]
                                )[0]
                            )
                else: 
                    xList.append(xn.reshape((-1,3)))

            if RIIS_on:
                xn[:] = doRIIS(xList, dxList, dim = RIIS_dim)
            else:
                xn += dx
                
            norm = N.dot(dx, dx)
            self.convergence = n, N.sqrt(norm/self.nx)
            # remove old data not needed anymore for RIIS
            if len(xList) >= RIIS_maxLength:
                dxList = dxList[-RIIS_maxLength:]
                xList = xList[-RIIS_maxLength:]

            if (norm < neps) and N.linalg.norm(dx, ord = N.inf) < maxEps:
                return self.convergence
            elif n >= maxiter:
                for i in range(len(self.xyz)//3):
                    ii = i*3
                    for j in range(i):
                        jj = j*3
                        d = N.linalg.norm(xn[ii:ii+3] - xn[jj:jj+3])
                        if d < 1.0:
                            warnings.warn(
                                "WARNING: Nuclear fusion of atoms " + 
                                "%i and %i is imminent (distance " %(i, j) +
                                "is %1.2f)." %d, Warning)
                print("Made %i steps and scaled %i of them" %(n, nScalings))
                raise ValueError('No convergence!')

        assert False, "This point shouldn't be reached, there is a bug."

    def sparseBackIteration(self, q, maxiter = 100, eps = 1e-6, initialize = 1,
            # iceps = 1e-6,  # old value (too small for RIIS)
            iceps = 1e-4, icp = 10, iclambda = 0.0001, maxStep = 0.5, warn = True,
            RIIS = True, RIIS_start = 20, RIIS_maxLength = 6, RIIS_dim = 4, 
            dampingThreshold = 0.1, maxEps = 1e-6,
            col_constraints = [], row_constraints=[]):
        """see: J. Chem. Phys. 113 (2000), 5598"""
        assert isinstance(maxStep, float)

        if warn: printWarning = self._printWarning
        else:    printWarning = lambda *x: None
        

        self.biInit()
        icArrays = self.tmp_icArrays

        # self.q = q # DEBUGGING
        # self.dx = [] # DEBUGGING

        dq = self.tmp_q
        dx = self.tmp_x
        xn = self.xyz
        if initialize: self.initA()

        # we call it once here, so we just need to update the values in the loop
        self.evalBt()   
        n = 0
        nScalings = 0 
        neps = eps*eps*self.nx
        xList = [] # geoms for RIIS
        dxList = [] # error vectors for RIIS
        RIIS_on = False

        normalizeIC(q, *icArrays)

        if self.torsions is not None: torsions = self.torsions
        else:                         torsions = None

        while 1:
            n += 1
            qn = self()
            if RIIS: 
                if not RIIS_on and n >= RIIS_start:
                    RIIS_on = True
                    # print("Turning on RIIS in step %i" %n)
            dq = N.subtract(q, qn, dq)

            # check phase 
            if torsions is not None: dq = ic.dphi_mod_2pi(dq, torsions)

            # step width control
            dqMax = N.max(N.abs(dq))
            if dqMax > maxStep:
                # print("scaling step %i" %n)
                nScalings += 1
                dq *= maxStep/dqMax

            self.evalB()

            # check all coordinates, if they are close to a singularity
            # i.e. stretches -> 0. , bends -> pi, bends of tors -> 0, pi
            # out of plains -> -pi/2, pi/2 or bends of out of plains -> 0., pi
            # If they get close, damp them away with 
            sparseDampIC(printWarning, qn, dq, self.B, dampingThreshold, 
                                                                    *icArrays)

            Bt = self.evalBt(update=1, perm=0)
            self.evalA()
            info = self.cholesky(eps=iceps, p=icp, lambd=iclambda)
            if info < 0:
                raise ArithmeticError(
                    'Incomplete Cholesky decomposition failed: ' + `info`)

            dx = self.inverseA(Bt(dq)).flatten()
            tmp = dx.copy()
            for c in col_constraints:
                tmp[c] = 0.0
            dx = tmp
            # self.dx.append(dx.copy()) # for DEBUGGING

            if RIIS and n >= RIIS_start - RIIS_maxLength:
                dxList.append(dx.copy().reshape((-1,3))) # store dx for RIIS
                if len(xList) > 0:
                    xList.append(
                        rigidBodySuperposition(
                            xn.reshape((-1,3)), xList[-1]
                                )[0]
                            )
                else: 
                    xList.append(xn.reshape((-1,3)))

            if RIIS_on:
                # print "doing RIIS step"
                xn[:] = doRIIS(xList, dxList, dim = RIIS_dim)
            else:
                xn += dx

            norm = N.dot(dx, dx)
            self.convergence = n, N.sqrt(norm/self.nx)
            # remove old data not needed anymore for RIIS
            if len(xList) >= RIIS_maxLength:
                dxList = dxList[-RIIS_maxLength:]
                xList = xList[-RIIS_maxLength:]


            if (norm < neps) and N.linalg.norm(dx, ord = N.inf) < maxEps:
                #print 'Back iteration converged after %i steps.' %n
                return self.convergence
            elif n >= maxiter:
                for i in range(len(self.xyz)//3):
                    ii = i*3
                    for j in range(i):
                        jj = j*3
                        d = N.linalg.norm(xn[ii:ii+3] - xn[jj:jj+3])
                        if d < 1.0:
                            warnings.warn(
                                "WARNING: Nuclear fusion of atoms " + 
                                "%i and %i is imminent (distance " %(i, j) +
                                "is %1.2f)." %d, Warning)
                print("Made %i steps and scaled %i of them" %(n, nScalings))
                raise ValueError('No convergence!')

        assert False, "This point shouldn't be reached, there is a bug."

    backIteration = sparseBackIteration
    bi = backIteration
    
    
    def denseinternalGradient(self, gx, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
   
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx, nxFloat)
        if not hasattr(self, 'gi'): self.gi = N.zeros(self.n, nxFloat)
        dg = self.tmp_q
        x = self.tmp_x
        gi = self.gi
        if initialize: self.initA()
        B = self.evalB(sort=0).full()
        Bt = self.evalBt(update=1,perm=0).full()
        At = N.dot(Bt,B)
        Atinv = regularizedInverse(At,eps=iceps).transpose()

        n = 0
        neps = eps*eps*self.nx
        while 1:
            n += 1
            x = N.dot(Bt,gi)
            dx = gx-x
            # print n
            # print x, dx
            norm = N.dot(dx, dx)
            # print norm
            dg = N.dot(B,N.dot(Atinv,dx)) 
            gi += dg
            if n > maxiter-1 or norm < neps: break
        # print gi
        # self.gi = gi
        return n, N.sqrt(norm/self.nx)


    def sparseinternalGradient(self, gx, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
        """see: J. Chem. Phys. 113 (2000), 5598"""
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx, nxFloat)
        if not hasattr(self, 'gi'): self.gi = N.zeros(self.n, nxFloat)
        dg = self.tmp_q
        x = self.tmp_x
        gi = self.gi
        if initialize: self.initA()
        B = self.evalB()
        Bt = self.evalBt(update=1, perm=0)
        A = self.evalA()
        info = self.cholesky(eps=iceps, p=icp, lambd=iclambda)
        if info < 1:
            raise ArithmeticError('Incomplete Cholesky decomposition failed: '
                   + `info`)
        # g0 = copy(x)
        # g0 = self.inverseA(B(gx))
        # gi = copy(g0)
        n = 0
        neps = eps*eps*self.nx
        while 1:
            n += 1
            x = Bt(gi)
            dx = gx-x
            print n
            # print x, dx
            norm = N.dot(dx, dx)
            print norm
            # dg = B(self.inverseA(dx))
            # dg = self.inverseA(B(dx))
            gprime = self.inverseLt(dx)
            gprime2 = self.inverseL(gprime)
            dg = B(gprime2)
            gi += dg
            if n > maxiter-1 or norm < neps: break
        # print gi
        # self.gi = gi
        return n, N.sqrt(norm/self.nx)

    # ig = sparseinternalGradient
    ig = denseinternalGradient

    def cartesianGradient(self, gi, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
        """
        transforms internal to cartesian gradient, self-consistently
        """
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx, nxFloat)
        if not hasattr(self, 'gx'): self.gx = N.zeros(self.nx, nxFloat)
        dg = self.tmp_x
        q = self.tmp_q
        gx = self.gx
        if initialize: self.initA()
        B = self.evalB()
        Bt = self.evalBt()
        A = self.evalA()
        self.cholesky(eps=iceps, p=icp, lambd=iclambda)
        gx =Bt(gi)
        n = 0
        neps = eps*eps*self.n
        while 1:
            n += 1
            q = B(self.inverseA(gx))
            diff = gi - q
            norm = N.dot(diff,diff)
            #print norm
            gx -= Bt(diff)
            if n > maxiter-1 or norm < neps: break
        return n, N.sqrt(norm/self.n)

    xg = cartesianGradient


class Periodic_icSystem(icSystem):
    """
    Periodic Boundary condition version of icSystem. 
    There are no translation, rotation B matrix components, 
    but there is a Bmatrix part for the lattice vectors

    Coordinates and lattice vectors are both cartesian absolute.
    """

    def __init__(self, intcrd, natom, xyz = None, \
            masses = None, cell = None, expand=1):

        if intcrd[-1] > 0:
            intcrd = list(intcrd)
            intcrd.append(0)
        self.ic = N.array(intcrd).astype(nxInt32)
        self.atoms_in_type = (2, 3, 4, 4, 1, 1, 1)
        self.natom = natom
        self.nx = 0
        if xyz is not None:
            self.xyz = N.ravel(xyz).astype(nxFloat)
            self.nx = len(self.xyz)
            if self.nx != 3*self.natom:
                raise IndexError("Array dimension of xyz does not match natom")
        else: self.xyz = xyz
        self.cell = cell
        self.celli = N.linalg.inv(cell)
        
        #get Bmatrix dimensions and number of DoFs Bmatrix has n times Bnnz
        # we add the three cell vectors as 9 rows at the end of the B matrix
        #self.Bnnz, self.n = ic.Bmatrix_pbc2_nnz(self.nx, self.ic)
        self.Bnnz, self.n = ic.Bmatrix_pbc_nnz(self.nx, self.ic)
        if expand==2:
            self.cell_indices = N.array([
            [0,0,0],[1,0,0],[0,1,0],[0,0,1],
            [1,1,0],[1,0,1],[0,1,1],[1,1,1],
            [-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,0],
            [-1,0,-1],[0,-1,-1],[1,-1,0],[-1,1,0],
            [1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1],
            [-1,1,1],[1,-1,1],[-1,-1,1],[1,1,-1],
            [-1,1,-1],[1,-1,-1],[-1,-1,-1],
            ])
        else:
            self.cell_indices = N.array([
                [0,0,0],[1,0,0],[0,1,0],[0,0,1],
                [1,1,0],[1,0,1],[0,1,1],[1,1,1]
                ])

        #duplicate coordinates
        xyz_pbc = N.empty([0,3])
        xyz = xyz.reshape(-1,3)
        for vec in self.cell_indices:
            trans = N.dot(vec,self.cell)
            xyz_tmp = xyz + trans
            xyz_pbc = N.concatenate([xyz_pbc, xyz_tmp]) 

        if xyz_pbc is not None:
            self.xyz_pbc = N.ravel(xyz_pbc.astype(nxFloat))
        else:
            self.xyz_pbc = xyz_pbc
        if cell is not None:
            self.cell = N.ravel(cell.astype(nxFloat))
        else: self.cell = N.ravel(N.eye(3).astype(nxFloat))
        self.celli = N.ravel(self.celli.astype(nxFloat))
        if masses is not None:
            if len(masses) != self.natom:
                raise IndexError("Array dimension of masses does not match natom")
            self.masses = N.array(masses).astype(nxFloat)
        else: self.masses = masses
        self.typeList = self.types()
        if len(self.typeList[2]) > 0:
            # function 'dphi_mod_2pi' of C module '_intcrd' needs an int32 array
            self.torsions = N.array(self.typeList[2], dtype = N.int32)[:,0]
        else:
            self.torsions = None
        self.initA()
    
    def initA(self):
        #self.connectivity()
        self.evalB(sort= 1)
        self.evalBt(perm = 0)
        #self.colamd(inverse=1)
        #self.evalBt()
    
    def evalB(self, sort = 0):
        if not hasattr(self, 'B'):
            self.B = CSR(n=self.n, m=self.nx+9, nnz=self.Bnnz, type = nxFloat)
        #ic.Bmatrix_pbc2(self.nx, self.xyz_pbc, self.ic, self.cell, self.celli, 
        ic.Bmatrix_pbc(self.nx, self.xyz_pbc, self.ic, 
               self.B.x, self.B.j, self.B.i, sort)

        return self.B

    def connectivity(self):
        self.cj, self.ci = ic.symbolicAc(self.ic, 8*self.natom)

    def __call__(self, x = None, cell = None):
        if x is None: x = self.xyz
        if x is None: return None
        if cell is None: cell = self.cell 
        #duplicate coordinates
        cell = cell.reshape(-1,3)
        xyz_pbc = N.empty([0,3])
        xyz = x.reshape(-1,3)
        for vec in self.cell_indices:
            trans = N.dot(vec,cell)
            xyz_tmp = xyz + trans
            xyz_pbc = N.concatenate([xyz_pbc, xyz_tmp]) 
        self.ic = N.ravel(self.ic.astype(nxInt32))
        xyz_pbc = N.ravel(xyz_pbc.astype(nxFloat))
        if not hasattr(self, 'crd'): self.crd = N.zeros(self.n, nxFloat)
        #ic.internals_pbc(xyz_pbc, self.celli, self.ic, self.crd)
        ic.internals(xyz_pbc, self.ic, self.crd)
        return self.crd
    
    def getq(self, x = None):
        if x is None: return None
        self.ic = N.ravel(self.ic.astype(nxInt32))
        xyz_pbc = N.ravel(x.astype(nxFloat))
        if not hasattr(self, 'crd'): self.crd = N.zeros(self.n, nxFloat)
        ic.internals(xyz_pbc, self.ic, self.crd)
        return self.crd
    
    def biInit(self):
        if not hasattr(self, 'tmp_cell'): self.tmp_cell = N.zeros(9, nxFloat)
        icSystem.biInit(self)
        self.tmp_x = N.zeros(self.nx+9, nxFloat)
    
    def denseBackIteration(self, q, maxiter = 200, eps = 1e-5, initialize = 1,
            inveps = 1e-9, maxStep = 0.5, warn = True, 
            RIIS = True, RIIS_start = 20, RIIS_maxLength = 6, RIIS_dim = 4, 
            restart = True, dampingThreshold = 0.1, maxEps = 1e-5,
            col_constraints = [], row_constraints=[],
            ):
        """
        Find Cartesian coordinates belonging to the set of internal coordinates
        'q' in an iterative way. 
        Parameters are:
        'inveps'         Parameter for the regularized inverse.
        'maxStep'        Maximal step during iteration. (Don't make 'maxStep'
                         too small, when dealing with convergence problems, 
                         try to increase the 'RIIS_dim' instead, since
                         RIIS will only work properly if steps are not scaled 
                         down. But also don't make maxStep too small, 
                         since then you might converge on spurious solutions.
        'RIIS'           Perform Regularized Inversion of the Iterative Subspace
                         to accelerate convergence.
        'RIIS_start'     Start RIIS after given number of steps.
        'RIIS_maxLength' maximal length of spanning set for the iterative 
                         subspace.
        'RIIS_dim'       Parameter for the Tikhonov regularization, so 
                         indirectly the dimension of space that will be 
                         inverted is determined here.
        """
        assert isinstance(maxStep, float)
        assert maxStep > eps
        self.biInit()
        icArrays = self.tmp_icArrays

        if warn: printWarning = self._printWarning
        else:    printWarning = lambda *x: None

        # self.q = q # DEBUGGING
        # self.dx = [] # DEBUGGING
        # bendsInOthers = self.getBendsInOthers() # DEBUGGING
        # ((stretch, bend, tors, oop), tmp) = self.getStretchBendTorsOop() # DEBUGGING

        dq = self.tmp_q
        dx = self.tmp_x
        #fracs = N.dot(self.xyz.reshape(-1,3),self.celli.reshape(-1,3)).flatten()
        #xn = N.concatenate([fracs, self.cell], axis=0)
        xn = N.concatenate([self.xyz, self.cell], axis=0)
        n = 0
        nScalings = 0 
        neps = eps*eps*(self.nx+9)
        xList = [] # geoms for RIIS
        dxList = [] # error vectors for RIIS
        RIIS_on = False

        normalizeIC(q, *icArrays)

        if self.torsions is not None: torsions = N.asarray(self.torsions, 
                                                            dtype = N.int32)
        else:
            torsions = None

        while 1:
            n += 1
            qn = self()
            # self.qn = qn # DEBUGGING

            # determine new dq by RIIS or traditional 
            # RIIS will be performed, if norm of last error vector is 
            # larger than the one of the second last vector and a sufficient 
            # number of vectors is available to do an extrapolation
            dq = N.subtract(q, qn, dq)
           
            if RIIS: 
                if not RIIS_on and n >= RIIS_start:
                    RIIS_on = True
                    # print("Turning on RIIS in step %i" %n)
            # check phase 
            if torsions is not None: dq = ic.dphi_mod_2pi(dq, torsions)

            # step width control
            dqMax = N.max(N.abs(dq))
            if dqMax > maxStep:
                # print("scaling step %i" %n)
                nScalings += 1
                dq *= maxStep/dqMax

            B = self.B.full()
            # self.Bfull = B # DEBUGGING

            # check all coordinates, if they are close to a singularity
            # i.e. stretches -> 0. , bends -> pi, bends of tors -> 0, pi
            # out of plains -> -pi/2, pi/2 or bends of out of plains -> 0., pi
            # If they get close, damp them away with 
            denseDampIC(printWarning, qn, dq, B, dampingThreshold, *icArrays)

            Bt = self.evalBt(update=1,perm=0).full()
            A = N.dot(Bt, B)
            self.Ainv = regularizedInverse(A, eps = inveps)
            
            #this dx is the displacement vector in x and the 3 lattice vectors
            dx = N.dot(self.Ainv, N.dot(N.transpose(B), dq))
            
            ##imposing internal constraints
            #c = []
            #for constraint in row_constraints:
                #c.append(B[constraint])
            #c = N.array(c).transpose()
            #from scipy.linalg import qr
            #(aa,bb) = qr(c,mode='full')
            #cc = aa.transpose()
            #for i,c in enumerate(row_constraints): 
                #e =  cc[c]#/N.linalg.norm(cc[c])
                #b = e * N.column_stack(e)
                #dx -= N.dot(b, dx)
            #imposing cartesian constraints
            for c in col_constraints:
                dx[c] = 0.0
            # self.dx.append(dx.copy()) # for DEBUGGING

            if RIIS and n >= RIIS_start - RIIS_maxLength:
                dxList.append(dx.copy().reshape((-1,3))) # store dx for RIIS
                if len(xList) > 0:
                    xList.append(
                        rigidBodySuperposition(
                            xn.reshape((-1,3)), xList[-1]
                                )[0]
                            )
                else: 
                    xList.append(xn.reshape((-1,3)))

            if RIIS_on:
                xn[:] = doRIIS(xList, dxList, dim = RIIS_dim)
            else:
                xn += dx
      
            self.xyz = xn[:-9]
            #self.xyz = N.dot(xn[:-9].reshape(-1,3),self.cell.reshape(-1,3)).flatten()
            self.cell = xn[-9:]
            #self.celli = N.linalg.inv(self.cell.reshape(-1,3)).flatten()
            norm = N.dot(dx, dx)
            self.convergence = n, N.sqrt(norm/(self.nx+9))
            # remove old data not needed anymore for RIIS
            if len(xList) >= RIIS_maxLength:
                dxList = dxList[-RIIS_maxLength:]
                xList = xList[-RIIS_maxLength:]

            if (norm < neps) and N.linalg.norm(dx, ord = N.inf) < maxEps:
                return self.convergence
            elif n >= maxiter:
                for i in range(len(self.xyz)//3):
                    ii = i*3
                    for j in range(i):
                        jj = j*3
                        d = N.linalg.norm(xn[ii:ii+3] - xn[jj:jj+3])
                        if d < 1.0:
                            warnings.warn(
                                "WARNING: Nuclear fusion of atoms " + 
                                "%i and %i is imminent (distance " %(i, j) +
                                "is %1.2f)." %d, Warning)
                print("Made %i steps and scaled %i of them" %(n, nScalings))
                raise ValueError('No convergence!')

        assert False, "This point shouldn't be reached, there is a bug."

    def sparseBackIteration(self, q, maxiter = 100, eps = 1e-6, initialize = 1,
             #iceps = 1e-6,  # old value (too small for RIIS)
            iceps = 1e-4, icp = 10, iclambda = 0.0001, maxStep = 0.5, warn = True,
            RIIS = True, RIIS_start = 20, RIIS_maxLength = 6, RIIS_dim = 4, 
            dampingThreshold = 0.1, maxEps = 1e-6,
            col_constraints = [], row_constraints=[]):
        """see: J. Chem. Phys. 113 (2000), 5598"""
        assert isinstance(maxStep, float)

        if warn: printWarning = self._printWarning
        else:    printWarning = lambda *x: None

        self.biInit()
        icArrays = self.tmp_icArrays

        # self.q = q # DEBUGGING
        # self.dx = [] # DEBUGGING

        dq = self.tmp_q
        dx = self.tmp_x
        #fracs = N.dot(self.xyz.reshape(-1,3),self.celli.reshape(-1,3)).flatten()
        #xn = N.concatenate([fracs, self.cell], axis=0)
        xn = N.concatenate([self.xyz, self.cell], axis=0)
        dx = N.concatenate([self.tmp_x, self.tmp_cell], axis=0)
        if initialize:self.initA()

        # we call it once here, so we just need to update the values in the loop
        self.evalBt(perm=0)   
        n = 0
        nScalings = 0 
        neps = eps*eps*self.nx
        xList = [] # geoms for RIIS
        dxList = [] # error vectors for RIIS
        RIIS_on = False

        normalizeIC(q, *icArrays)

        if self.torsions is not None: torsions = self.torsions
        else:                         torsions = None

        while 1:
            n += 1
            qn = self()
            if RIIS: 
                if not RIIS_on and n >= RIIS_start:
                    RIIS_on = True
                    # print("Turning on RIIS in step %i" %n)

            dq = N.subtract(q, qn, dq)
            
            #print 'qn ', qn[:8]
            #print 'q ', q[:8]
            # check phase 
            if torsions is not None: dq = ic.dphi_mod_2pi(dq, torsions)

            # step width control
            dqMax = N.max(N.abs(dq))
            if dqMax > maxStep:
                # print("scaling step %i" %n)
                nScalings += 1
                dq *= maxStep/dqMax

            self.evalB()

            # check all coordinates, if they are close to a singularity
            # i.e. stretches -> 0. , bends -> pi, bends of tors -> 0, pi
            # out of plains -> -pi/2, pi/2 or bends of out of plains -> 0., pi
            # If they get close, damp them away with 
            sparseDampIC(printWarning, qn, dq, self.B, dampingThreshold, 
                                                                    *icArrays)

            Bt = self.evalBt(update=1, perm=0)
            self.evalA()
            info = self.cholesky(eps=iceps, p=icp, lambd=iclambda)
            if info < 0:
                raise ArithmeticError(
                    'Incomplete Cholesky decomposition failed: ' + `info`)
            dx = self.inverseA(Bt(dq))
            # self.dx.append(dx.copy()) # for DEBUGGING
            
            ##imposing cartesian constraints
            tmp = dx.copy()
            for c in col_constraints:
                tmp[c] = 0
            dx = tmp
            #print 'dx ', tmp[:8]
            if RIIS and n >= RIIS_start - RIIS_maxLength:
                dxList.append(dx.copy().reshape((-1,3))) # store dx for RIIS
                if len(xList) > 0:
                    xList.append(
                        rigidBodySuperposition(
                            xn.reshape((-1,3)), xList[-1]
                                )[0]
                            )
                else: 
                    xList.append(xn.reshape((-1,3)))

            if RIIS_on:
                # print "doing RIIS step"
                xn[:] = doRIIS(xList, dxList, dim = RIIS_dim)
            else:
                xn += dx 
            
            self.xyz = xn[:-9]
            #self.xyz = N.dot(xn[:-9].reshape(-1,3),self.cell.reshape(-1,3)).flatten()
            self.cell = xn[-9:]
            #self.celli = N.linalg.inv(self.cell.reshape(-1,3)).flatten()
            norm = N.dot(dx, dx)
            #print 'norm ', norm, neps
            self.convergence = n, N.sqrt(norm/self.nx+9)
            # remove old data not needed anymore for RIIS
            if len(xList) >= RIIS_maxLength:
                dxList = dxList[-RIIS_maxLength:]
                xList = xList[-RIIS_maxLength:]


            if (norm < neps) and N.linalg.norm(dx, ord = N.inf) < maxEps:
                #print 'Back iteration converged after %i steps.' %n
                return self.convergence
            elif n >= maxiter:
                for i in range(len(self.xyz)//3):
                    ii = i*3
                    for j in range(i):
                        jj = j*3
                        d = N.linalg.norm(self.xyz[ii:ii+3] - self.xyz[jj:jj+3])
                        if d < 1.0:
                            warnings.warn(
                                "WARNING: Nuclear fusion of atoms " + 
                                "%i and %i is imminent (distance " %(i, j) +
                                "is %1.2f)." %d, Warning)
                print("Made %i steps and scaled %i of them" %(n, nScalings))
                raise ValueError('No convergence!')

        assert False, "This point shouldn't be reached, there is a bug."
    
    backIteration = sparseBackIteration
    bi = backIteration
    
    def internalGradient(self, gx, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
        """see: J. Chem. Phys. 113 (2000), 5598"""
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx+9, nxFloat)
        if not hasattr(self, 'gi'): self.gi = N.zeros(self.n, nxFloat)
        dg = self.tmp_q
        x = self.tmp_x
        gi = self.gi
        if initialize: self.initA()
        B = self.evalB(sort=1)
        Bt = self.evalBt(perm=0)
        A = self.evalA()
        info = self.cholesky(eps=iceps, p=icp, lambd=iclambda)
        if info < 1:
            raise ArithmeticError('Incomplete Cholesky decomposition failed: '
                    + `info`)
        g0 = copy(x)
        gi = B(self.inverseA(gx))
        n = 0
        neps = eps*eps*self.nx
        while 1:
            n += 1
            x = Bt(gi, x)
            x -= g0
            norm = N.dot(x, x)
            #print norm
            dg = B(self.inverseA(x), dg)
            gi -= dg
            if n > maxiter-1 or norm < neps: break
        return n, N.sqrt(norm/self.nx)

    ig = internalGradient
    
    def cartesianGradient(self, gi, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
        """
        transforms internal to cartesian gradient, self-consistently
        """
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx+9, nxFloat)
        if not hasattr(self, 'gx'): self.gx = N.zeros(self.nx+9, nxFloat)
        dg = self.tmp_x
        q = self.tmp_q
        gx = self.gx
        if initialize: self.initA()
        B = self.evalB()
        Bt = self.evalBt()
        A = self.evalA()
        self.cholesky(eps=iceps, p=icp, lambd=iclambda)
        gx =Bt(gi)
        n = 0
        neps = eps*eps*self.n
        while 1:
            n += 1
            q = B(self.inverseA(gx))
            diff = gi - q
            norm = N.dot(diff,diff)
            #print norm
            gx += Bt(diff)
            if n > maxiter-1 or norm < neps: break
        self.gx = gx
        return n, N.sqrt(norm/self.n)

    xg = cartesianGradient

def doRIIS(x, e, dim = 3):
    """
    Do an Regularized Inversion of the Iterative Subspace (RIIS) for each atom.
    'dim' gives an approximate cutoff for the Tikhonov regularization.
    HINT: This could be optimized by saving the correlation matrices M for 
    each atom and calculating only the new vector after each iteration step.
    """
    assert len(e) == len(x)
    # order the error vectors and positions of the atoms so, that there is
    # a list of those vectors for each atom.
    X = N.rollaxis(N.asarray(x), 1,0)
    E = N.rollaxis(N.asarray(e), 1,0)
    Xn = N.empty(x[0].shape) # the new interpolated geometry
    n = len(e) + 1
    M = -N.ones((n,n) ) # correlation matrix
    b = N.zeros(n) # inhomogeneity of the linear equation system
    w = N.zeros(n) # interpolation weights
    M[-1,-1] = 0
    b[-1] = -1
    for (x, e, xn) in zip(X, E, Xn):
        for i in xrange(len(e)):
            for j in range(i+1):
                M[i,j] = M[j,i] = N.dot(e[i], e[j])
        # do Tikhonov regularization
        U, s, VT = N.linalg.svd(M)
        eps = s[dim] # eps is determined by the singular values 
        s2 = s*s 
        s2 += eps*eps
        s /= s2
        Mi = N.dot(U, s[:,N.newaxis]*VT) # regularized inverse
        w = N.dot(Mi, b) # approximate weights
        xn[:] = sum( w_i*x_i for (w_i, x_i) in zip(w[:-1], x))
    return Xn.flatten()
    
    

class Periodic_icSystem2(Periodic_icSystem):
    """
    Periodic Boundary condition version of icSystem. 
    There are no translation, rotation B matrix components, 
    but there is a Bmatrix part for the lattice vectors

    Coordinates are in fractional coordinates wrt to unit cell
    Lattice vectors are absolute.
    """

    def __init__(self, intcrd, natom, xyz = None, \
            masses = None, cell = None, expand=1):
        if intcrd[-1] > 0:
            intcrd = list(intcrd)
            intcrd.append(0)
        self.ic = N.array(intcrd).astype(nxInt32)
        self.atoms_in_type = (2, 3, 4, 4, 1, 1, 1)
        self.natom = natom
        self.nx = 0
        if xyz is not None:
            self.xyz = N.ravel(xyz).astype(nxFloat)
            self.nx = len(self.xyz)
            if self.nx != 3*self.natom:
                raise IndexError("Array dimension of xyz does not match natom")
        else: self.xyz = xyz
        self.cell = cell
        self.celli = N.linalg.inv(cell)
        
        #get Bmatrix dimensions and number of DoFs Bmatrix has n times Bnnz
        # we add the three cell vectors as 9 rows at the end of the B matrix
        self.Bnnz, self.n = ic.Bmatrix_pbc2_nnz(self.nx, self.ic)
        if expand==2:
            self.cell_indices = N.array([
            [0,0,0],[1,0,0],[0,1,0],[0,0,1],
            [1,1,0],[1,0,1],[0,1,1],[1,1,1],
            [-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,0],
            [-1,0,-1],[0,-1,-1],[1,-1,0],[-1,1,0],
            [1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1],
            [-1,1,1],[1,-1,1],[-1,-1,1],[1,1,-1],
            [-1,1,-1],[1,-1,-1],[-1,-1,-1],
            ])
        else:
            self.cell_indices = N.array([
                [0,0,0],[1,0,0],[0,1,0],[0,0,1],
                [1,1,0],[1,0,1],[0,1,1],[1,1,1]
                ])

        #duplicate coordinates
        xyz_pbc = N.empty([0,3])
        xyz = xyz.reshape(-1,3)
        for vec in self.cell_indices:
            trans = N.dot(vec,self.cell)
            xyz_tmp = xyz + trans
            xyz_pbc = N.concatenate([xyz_pbc, xyz_tmp]) 

        if xyz_pbc is not None:
            self.xyz_pbc = N.ravel(xyz_pbc.astype(nxFloat))
        else:
            self.xyz_pbc = xyz_pbc
        if cell is not None:
            self.cell = N.ravel(cell.astype(nxFloat))
        else: self.cell = N.ravel(N.eye(3).astype(nxFloat))
        self.celli = N.ravel(self.celli.astype(nxFloat))
        if masses is not None:
            if len(masses) != self.natom:
                raise IndexError("Array dimension of masses does not match natom")
            self.masses = N.array(masses).astype(nxFloat)
        else: self.masses = masses
        self.typeList = self.types()
        if len(self.typeList[2]) > 0:
            # function 'dphi_mod_2pi' of C module '_intcrd' needs an int32 array
            self.torsions = N.array(self.typeList[2], dtype = N.int32)[:,0]
        else:
            self.torsions = None
        self.initA()
    
    def evalB(self, sort = 0):
        if not hasattr(self, 'B'):
            self.B = CSR(n=self.n, m=self.nx+9, nnz=self.Bnnz, type = nxFloat)
        ic.Bmatrix_pbc2(self.nx, self.xyz_pbc, self.ic, self.cell, self.celli, 
               self.B.x, self.B.j, self.B.i, sort)
        return self.B
    
    def __call__(self, x = None, cell = None):
        if x is None: x = self.xyz
        if x is None: return None
        if cell is None: cell = self.cell 
        #duplicate coordinates
        cell = cell.reshape(-1,3)
        xyz_pbc = N.empty([0,3])
        xyz = x.reshape(-1,3)
        for vec in self.cell_indices:
            trans = N.dot(vec,cell)
            xyz_tmp = xyz + trans
            xyz_pbc = N.concatenate([xyz_pbc, xyz_tmp]) 
        self.ic = N.ravel(self.ic.astype(nxInt32))
        xyz_pbc = N.ravel(xyz_pbc.astype(nxFloat))
        if not hasattr(self, 'crd'): self.crd = N.zeros(self.n, nxFloat)
        ic.internals_pbc(xyz_pbc, self.celli, self.ic, self.crd)
        return self.crd
    
    def denseBackIteration(self, q, maxiter = 200, eps = 1e-5, initialize = 1,
            inveps = 1e-9, maxStep = 0.5, warn = True, 
            RIIS = True, RIIS_start = 20, RIIS_maxLength = 6, RIIS_dim = 4, 
            restart = True, dampingThreshold = 0.1, maxEps = 1e-5,
            col_constraints = [], row_constraints=[],
            ):
        """
        Find Cartesian coordinates belonging to the set of internal coordinates
        'q' in an iterative way. 
        Parameters are:
        'inveps'         Parameter for the regularized inverse.
        'maxStep'        Maximal step during iteration. (Don't make 'maxStep'
                         too small, when dealing with convergence problems, 
                         try to increase the 'RIIS_dim' instead, since
                         RIIS will only work properly if steps are not scaled 
                         down. But also don't make maxStep too small, 
                         since then you might converge on spurious solutions.
        'RIIS'           Perform Regularized Inversion of the Iterative Subspace
                         to accelerate convergence.
        'RIIS_start'     Start RIIS after given number of steps.
        'RIIS_maxLength' maximal length of spanning set for the iterative 
                         subspace.
        'RIIS_dim'       Parameter for the Tikhonov regularization, so 
                         indirectly the dimension of space that will be 
                         inverted is determined here.
        """
        assert isinstance(maxStep, float)
        assert maxStep > eps
        self.biInit()
        icArrays = self.tmp_icArrays

        if warn: printWarning = self._printWarning
        else:    printWarning = lambda *x: None

        # self.q = q # DEBUGGING
        # self.dx = [] # DEBUGGING
        # bendsInOthers = self.getBendsInOthers() # DEBUGGING
        # ((stretch, bend, tors, oop), tmp) = self.getStretchBendTorsOop() # DEBUGGING

        dq = self.tmp_q
        dx = self.tmp_x
        fracs = N.dot(self.xyz.reshape(-1,3),self.celli.reshape(-1,3)).flatten()
        xn = N.concatenate([fracs, self.cell], axis=0)
        n = 0
        nScalings = 0 
        neps = eps*eps*(self.nx+9)
        xList = [] # geoms for RIIS
        dxList = [] # error vectors for RIIS
        RIIS_on = False

        normalizeIC(q, *icArrays)

        if self.torsions is not None: torsions = N.asarray(self.torsions, 
                                                            dtype = N.int32)
        else:
            torsions = None

        while 1:
            n += 1
            qn = self()
            # self.qn = qn # DEBUGGING

            # determine new dq by RIIS or traditional 
            # RIIS will be performed, if norm of last error vector is 
            # larger than the one of the second last vector and a sufficient 
            # number of vectors is available to do an extrapolation
            dq = N.subtract(q, qn, dq)
           
            if RIIS: 
                if not RIIS_on and n >= RIIS_start:
                    RIIS_on = True
                    # print("Turning on RIIS in step %i" %n)
            # check phase 
            if torsions is not None: dq = ic.dphi_mod_2pi(dq, torsions)

            # step width control
            dqMax = N.max(N.abs(dq))
            if dqMax > maxStep:
                # print("scaling step %i" %n)
                nScalings += 1
                dq *= maxStep/dqMax

            B = self.B.full()
            # self.Bfull = B # DEBUGGING

            # check all coordinates, if they are close to a singularity
            # i.e. stretches -> 0. , bends -> pi, bends of tors -> 0, pi
            # out of plains -> -pi/2, pi/2 or bends of out of plains -> 0., pi
            # If they get close, damp them away with 
            denseDampIC(printWarning, qn, dq, B, dampingThreshold, *icArrays)

            Bt = self.evalBt(update=1,perm=0).full()
            A = N.dot(Bt, B)
            self.Ainv = regularizedInverse(A, eps = inveps)
            
            #this dx is the displacement vector in x and the 3 lattice vectors
            dx = N.dot(self.Ainv, N.dot(Bt, dq))
            
            ##imposing internal constraints
            #c = []
            #for constraint in row_constraints:
                #c.append(B[constraint])
            #c = N.array(c).transpose()
            #from scipy.linalg import qr
            #(aa,bb) = qr(c,mode='full')
            #cc = aa.transpose()
            #for i,c in enumerate(row_constraints): 
                #e =  cc[c]#/N.linalg.norm(cc[c])
                #b = e * N.column_stack(e)
                #dx -= N.dot(b, dx)
            #imposing cartesian constraints
            for c in col_constraints:
                dx[c] = 0.0
            # self.dx.append(dx.copy()) # for DEBUGGING

            if RIIS and n >= RIIS_start - RIIS_maxLength:
                dxList.append(dx.copy().reshape((-1,3))) # store dx for RIIS
                if len(xList) > 0:
                    xList.append(
                        rigidBodySuperposition(
                            xn.reshape((-1,3)), xList[-1]
                                )[0]
                            )
                else: 
                    xList.append(xn.reshape((-1,3)))

            if RIIS_on:
                xn[:] = doRIIS(xList, dxList, dim = RIIS_dim)
            else:
                xn += dx
      
            self.xyz = N.dot(xn[:-9].reshape(-1,3),self.cell.reshape(-1,3)).flatten()
            self.cell = xn[-9:]
            self.celli = N.linalg.inv(self.cell.reshape(-1,3)).flatten()
            norm = N.dot(dx, dx)
            self.convergence = n, N.sqrt(norm/(self.nx+9))
            # remove old data not needed anymore for RIIS
            if len(xList) >= RIIS_maxLength:
                dxList = dxList[-RIIS_maxLength:]
                xList = xList[-RIIS_maxLength:]

            if (norm < neps) and N.linalg.norm(dx, ord = N.inf) < maxEps:
                return self.convergence
            elif n >= maxiter:
                for i in range(len(self.xyz)//3):
                    ii = i*3
                    for j in range(i):
                        jj = j*3
                        d = N.linalg.norm(xn[ii:ii+3] - xn[jj:jj+3])
                        if d < 1.0:
                            warnings.warn(
                                "WARNING: Nuclear fusion of atoms " + 
                                "%i and %i is imminent (distance " %(i, j) +
                                "is %1.2f)." %d, Warning)
                print("Made %i steps and scaled %i of them" %(n, nScalings))
                raise ValueError('No convergence!')

        assert False, "This point shouldn't be reached, there is a bug."
    
    def sparseBackIteration(self, q, maxiter = 100, eps = 1e-6, initialize = 1,
             #iceps = 1e-6,  # old value (too small for RIIS)
            iceps = 1e-4, icp = 10, iclambda = 0.0001, maxStep = 0.5, warn = True,
            RIIS = True, RIIS_start = 20, RIIS_maxLength = 6, RIIS_dim = 4, 
            dampingThreshold = 0.1, maxEps = 1e-6,
            col_constraints = [], row_constraints=[]):
        """see: J. Chem. Phys. 113 (2000), 5598"""
        assert isinstance(maxStep, float)

        if warn: printWarning = self._printWarning
        else:    printWarning = lambda *x: None

        self.biInit()
        icArrays = self.tmp_icArrays

        # self.q = q # DEBUGGING
        # self.dx = [] # DEBUGGING

        dq = self.tmp_q
        dx = self.tmp_x
        fracs = N.dot(self.xyz.reshape(-1,3),self.celli.reshape(-1,3)).flatten()
        xn = N.concatenate([fracs, self.cell], axis=0)
        dx = N.concatenate([self.tmp_x, self.tmp_cell], axis=0)
        if initialize:self.initA()

        # we call it once here, so we just need to update the values in the loop
        self.evalBt(perm=0)   
        n = 0
        nScalings = 0 
        neps = eps*eps*self.nx
        xList = [] # geoms for RIIS
        dxList = [] # error vectors for RIIS
        RIIS_on = False

        normalizeIC(q, *icArrays)

        if self.torsions is not None: torsions = self.torsions
        else:                         torsions = None

        while 1:
            n += 1
            qn = self()
            if RIIS: 
                if not RIIS_on and n >= RIIS_start:
                    RIIS_on = True
                    # print("Turning on RIIS in step %i" %n)

            dq = N.subtract(q, qn, dq)
            
            #print 'qn ', qn[:8]
            #print 'q ', q[:8]
            # check phase 
            if torsions is not None: dq = ic.dphi_mod_2pi(dq, torsions)

            # step width control
            dqMax = N.max(N.abs(dq))
            if dqMax > maxStep:
                # print("scaling step %i" %n)
                nScalings += 1
                dq *= maxStep/dqMax

            self.evalB()

            # check all coordinates, if they are close to a singularity
            # i.e. stretches -> 0. , bends -> pi, bends of tors -> 0, pi
            # out of plains -> -pi/2, pi/2 or bends of out of plains -> 0., pi
            # If they get close, damp them away with 
            sparseDampIC(printWarning, qn, dq, self.B, dampingThreshold, 
                                                                    *icArrays)

            Bt = self.evalBt(update=1, perm=0)
            self.evalA()
            info = self.cholesky(eps=iceps, p=icp, lambd=iclambda)
            if info < 0:
                raise ArithmeticError(
                    'Incomplete Cholesky decomposition failed: ' + `info`)
            dx = self.inverseA(Bt(dq))
            # self.dx.append(dx.copy()) # for DEBUGGING
            
            ##imposing cartesian constraints
            tmp = dx.copy()
            for c in col_constraints:
                tmp[c] = 0
            dx = tmp
            #print 'dx ', tmp[:8]
            if RIIS and n >= RIIS_start - RIIS_maxLength:
                dxList.append(dx.copy().reshape((-1,3))) # store dx for RIIS
                if len(xList) > 0:
                    xList.append(
                        rigidBodySuperposition(
                            xn.reshape((-1,3)), xList[-1]
                                )[0]
                            )
                else: 
                    xList.append(xn.reshape((-1,3)))

            if RIIS_on:
                # print "doing RIIS step"
                xn[:] = doRIIS(xList, dxList, dim = RIIS_dim)
            else:
                xn += dx 
            
            self.cell = xn[-9:]
            self.xyz = N.dot(xn[:-9].reshape(-1,3),self.cell.reshape(-1,3)).flatten()
            #self.cell = xn[-9:]
            self.celli = N.linalg.inv(self.cell.reshape(-1,3)).flatten()
            norm = N.dot(dx, dx)
            #print 'norm ', norm
            self.convergence = n, N.sqrt(norm/self.nx+9)
            # remove old data not needed anymore for RIIS
            if len(xList) >= RIIS_maxLength:
                dxList = dxList[-RIIS_maxLength:]
                xList = xList[-RIIS_maxLength:]


            if (norm < neps) and N.linalg.norm(dx, ord = N.inf) < maxEps:
                # print 'Back iteration converged after %i steps.' %n
                return self.convergence
            elif n >= maxiter:
                for i in range(len(self.xyz)//3):
                    ii = i*3
                    for j in range(i):
                        jj = j*3
                        d = N.linalg.norm(self.xyz[ii:ii+3] - self.xyz[jj:jj+3])
                        if d < 1.0:
                            warnings.warn(
                                "WARNING: Nuclear fusion of atoms " + 
                                "%i and %i is imminent (distance " %(i, j) +
                                "is %1.2f)." %d, Warning)
                print("Made %i steps and scaled %i of them" %(n, nScalings))
                raise ValueError('No convergence!')

        assert False, "This point shouldn't be reached, there is a bug."
    
    backIteration = sparseBackIteration
    bi = backIteration
    
    def denseinternalGradient(self, gx, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
   
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx+9, nxFloat)
        if not hasattr(self, 'gi'): self.gi = N.zeros(self.n, nxFloat)
        dg = self.tmp_q
        x = self.tmp_x
        gi = self.gi
        if initialize: self.initA()
        B = self.evalB(sort=0).full()
        Bt = self.evalBt(update=1,perm=0).full()
        At = N.dot(Bt,B)
        Atinv = regularizedInverse(At,eps=iceps).transpose()
        #transform to fractional forces
        # gx = N.dot(gx,self.cell).flatten()

        n = 0
        neps = eps*eps*(self.nx+9)
        while 1:
            n += 1
            x = N.dot(Bt,gi)
            dx = gx-x
            # print n
            # print x, dx
            norm = N.dot(dx, dx)
            # print norm
            dg = N.dot(B,N.dot(Atinv,dx)) 
            gi += dg
            if n > maxiter-1 or norm < neps: break
        # print gi
        # self.gi = gi
        return n, N.sqrt(norm/self.nx)
    
    def internalGradient(self, gx, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
        """see: J. Chem. Phys. 113 (2000), 5598"""
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx+9, nxFloat)
        if not hasattr(self, 'gi'): self.gi = N.zeros(self.n, nxFloat)
        dg = self.tmp_q
        x = self.tmp_x
        gi = self.gi
        if initialize: self.initA()
        B = self.evalB(sort=1)
        Bt = self.evalBt(perm=0)
        A = self.evalA()
        info = self.cholesky(eps=iceps, p=icp, lambd=iclambda)
        if info < 1:
            raise ArithmeticError('Incomplete Cholesky decomposition failed: '
                    + `info`)
        g0 = copy(x)
        #transform to fractional forces
        gx = N.dot(gx,self.cell)
        gi = B(self.inverseA(gx))
        n = 0
        neps = eps*eps*self.nx
        while 1:
            n += 1
            x = Bt(gi, x)
            x -= g0
            norm = N.dot(x, x)
            # print norm
            dg = B(self.inverseA(x), dg)
            gi -= dg
            if n > maxiter-1 or norm < neps: break
        return n, N.sqrt(norm/self.nx)

    # ig = internalGradient
    ig = denseinternalGradient
    
    def cartesianGradient(self, gi, maxiter = 20, eps = 1e-6, initialize = 1,
            iceps = 1e-6, icp = 10, iclambda = 0.0001):
        """
        transforms internal to cartesian gradient, self-consistently
        """
        if not hasattr(self, 'tmp_q'): self.tmp_q = N.zeros(self.n, nxFloat)
        if not hasattr(self, 'tmp_x'): self.tmp_x = N.zeros(self.nx+9, nxFloat)
        if not hasattr(self, 'gx'): self.gx = N.zeros(self.nx+9, nxFloat)
        dg = self.tmp_x
        q = self.tmp_q
        gx = self.gx
        #if initialize: self.initA()
        B = self.evalB(sort=1)
        Bt = self.evalBt(perm=0)
        A = self.evalA()
        self.cholesky(eps=iceps, p=icp, lambd=iclambda)
        gx = Bt(gi)
        #transforming fractional coords into cartesian
        #gx[:-9] = np.dot(gx[:-9].reshape(-1,3), self.celli.reshape(-1,3)).flatten()
        n = 0
        neps = eps*eps*self.n
        while 1:
            n += 1
            q = B(self.inverseA(gx))
            diff = gi - q
            norm = N.dot(diff,diff)
            #print norm
            gx += Bt(diff) 
            if n > maxiter-1 or norm < neps: break
        gx[:-9] = N.dot(gx[:-9].reshape(-1,3), self.celli.reshape(-1,3)).flatten()
        self.gx = gx
        return n, N.sqrt(norm/self.n)

    xg = cartesianGradient
