# numeric.Rotation
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2004 Christoph Scheurer
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

"""
Rigid Body rotations
"""

from winak.curvilinear.numeric import *
from operator import itemgetter

def rotationFromQuaternion(e):
    s0 = e[0]*e[0]
    s1 = e[1]*e[1]
    s2 = e[2]*e[2]
    s3 = e[3]*e[3]
    e01 = e[0]*e[1]
    e02 = e[0]*e[2]
    e03 = e[0]*e[3]
    e12 = e[1]*e[2]
    e13 = e[1]*e[3]
    e23 = e[2]*e[3]
    return N.array(((s0+s1-s2-s3, 2*(e12+e03), 2*(e13-e02)),
                    (2*(e12-e03), s0-s1+s2-s3, 2*(e23+e01)),
                    (2*(e13+e02), 2*(e23-e01), s0-s1-s2+s3)))

def directionCosines(u, v, x, y, z):
    x1 = x.normal()
    x2 = y.normal()
    x3 = z.normal()
    v1 = u.normal()
    v3 = v1.cross(v).normal()
    v2 = v3.cross(v1).normal()
    return N.array(((x1*v1, x1*v2, x1*v3),
                    (x2*v1, x2*v2, x2*v3),
                    (x3*v1, x3*v2, x3*v3)))

def EulerParameter(u, v, x, y, z):
    """compute the Euler parameters (quaternions) [Goldstein (1980, ch. 4-5)]
       for transforming the right-handed coordinate system given by vectors
       (x, y, z) into the one (w1, w2, w3) defined by vectors u, v with the
       convention:
           v1 || u
           v2 in plane(u, v)
           v3 = v1 x v2
    """
    x1 = x.normal()
    x2 = y.normal()
    x3 = z.normal()
    v1 = u.normal()
    v3 = v1.cross(v).normal()
    v2 = v3.cross(v1).normal()
# algorithm see: http://vamos.sourceforge.net/matrixfaq.htm#Q48
    a11 = x1*v1
    a22 = x2*v2
    a33 = x3*v3
    t = a11 + a22 + a33 + 1
    if t > 0:
        print 0
        s = 2 * N.sqrt(t)
        e = (t, x2*v3 - x3*v2, x3*v1 - x1*v3, x1*v2 - x2*v1)
    else:
        q0 = x1*v2 + x2*v1
        q1 = x1*v3 + x3*v1
        q2 = x2*v3 + x3*v2
        if a11 > a22 and a11 > a33:
            print 1
            s = 2 * N.sqrt(1 + a11 - a22 - a33)
            e = (q2, 0.5, q0, q1)
        elif a22 > a33 and a22 > a11:
            print 2
            s = 2 * N.sqrt(1 - a11 + a22 - a33)
            e = (q1, q0, 0.5, q2)
        else:
            print 3
            s = 2 * N.sqrt(1 - a11 - a22 + a33)
            e = (q0, q2, q1, 0.5)
    return N.array(e)/s

def EulerAngle(u, v, x, y, z, eps = 1.0e-7):
    """compute the Euler angles phi, theta, psi in the so-called "x-convention"
       [Goldstein (1980, ch. 4-4) and Landau and Lifschitz (1976)]
       for transforming the right-handed coordinate system given by vectors
       (x, y, z) into the one (e1, e2, e3) defined by vectors u, v with the
       convention:
       e1 || u
       e2 in plane(u, v)
       e3 = e1 x e2
    """
    phi = 0.0
    theta = 0.0
    psi = 0.0
    ex = x / N.linalg.norm(x)
    ey = y / N.linalg.norm(y)
    ez = z / N.linalg.norm(z)
    e1 = u  / N.linalg.norm(u)
    e3 = N.cross(e1, v)
    e3 /= N.linalg.norm(e3)
    e2 = N.cross(e3,e1)
    e2 /= N.linalg.norm(e2)
    a33 = N.dot(ez,e3)    #   cos(theta)
    theta = N.arccos(a33)
    st = N.sin(theta)
    if abs(st) > eps:
        a13 = N.dot(ex, e3)    #   sin(psi) sin(theta)
        a23 = N.dot(ey, e3)    #   cos(psi) sin(theta)
        a31 = N.dot(ez, e1)    #   sin(phi) sin(theta)
        a32 = N.dot(ez, e2)    # - cos(phi) sin(theta)
        ps =   N.arcsin(a13/st)
        fs =   N.arcsin(a31/st)
        pc =   N.arccos(a23/st)
        fc = - N.arccos(a32/st)
    else:       # theta = n * pi
        pass
    return phi, theta, psi

#by REINI
def EulerAngles_LC2WC(x1, y1, z1):
    """Calculate the Euler angles in "x-convention" between
    the World Coordinates (1 0 0), (0 1 0), (0 0 1) and the
    local coordinate system defined by x1, y1, z1
    algorithm by Nikolaev -> www.geom3d.com
    """
    import sys
    epsilon = sys.float_info.epsilon

    x1 /= N.linalg.norm(x1)
    y1 /= N.linalg.norm(y1)
    z1 /= N.linalg.norm(z1)

    z1xy = N.sqrt(z1[0]*z1[0] + z1[1]*z1[1])

    if z1xy > epsilon:
        x_p = N.cross(z1,N.array([0.,0.,1.0]))
        x_px = N.dot(x_p,x1)
        x_py = N.dot(x_p,y1)
        phi = N.arctan2(y1[0]*z1[1]-y1[1]*z1[0], x1[0]*z1[1]-x1[1]*z1[0])
        theta = N.arctan2(z1xy,z1[2])
        psi = -N.arctan2(-z1[0],z1[1])
    else:
        phi = 0.0
        if z1[2]>0.0:
            theta = 0.0
        else:
            theta = N.pi
        psi = -N.arctan2(x1[1],x1[0])

    return phi, theta, psi

def EulerAngles_WC2LC(x1, y1, z1):
    """Euler Angles for going from WC to LC
    all in x-convention
    """

    phi, theta, psi = EulerAngles_LC2WC(x1, y1, z1)
    return -psi, -theta, -phi



"""
Rigid body superposition by Daniel Strobusch
"""

from Quaternions import Quaternion

def rigidBodySuperposition(X, Y, weights = None, RMSD = False, shift = True):
    """
    Align 'X' to 'Y' minimizing the RMSD of the coordinates.
    The parameter 'weights' specifies how the coordinates should be
    weighted during the alignment procedure. Possible options are:
    'None' : equal weight for all atoms
    array of weights : a user defined array of weights, e.g. to align only
    a fraction of the atoms by setting the weights for
    some atoms to zero or to use mass weighting
    'RMSD' defines if the RMSD should be computed.
    Returns a tuple of transformed X, the RMSD and a tuple defining the
    transformation, i.e a vector, a quaternion and another vector.
    To apply the transformation '(v, q, w)' to an arbitrary set of
    coordinates 'r' calculate q.rotate(r+v) + w
    Algorithm depending on:
    "Kneller, Gerald R. (1991)
    'Superposition of Molecular Structures using Quaternions',
    Molecular Simulation, 7: 1, 113 - 119"
    """
    X = X.copy()
    Y = Y.copy()
    assert X.ndim == 2
    assert X.shape == Y.shape
    n = len(X) # the number of atoms specified in the molecule
    if weights == None: # equal weights for all atoms
        W = N.array(N.ones(n))
    else: # use custom weights
        W = N.array(weights, dtype = N.float64)
        assert len(W) == len(X)
    W /= N.sum(W) # norm weights
    if shift:
        xC = sum( x*w for (x,w) in zip(X,W) ) # center of rotation
        yC = sum( y*w for (y,w) in zip(Y,W) ) # center of rotation
    else:
        xC = N.zeros(3)
        yC = N.zeros(3)
    # shift centers of rotation to origin
    X -= xC
    Y -= yC
    M = N.zeros((4,4), dtype = N.float) # empty matrix to determine quaternion
    xy = N.empty(3, dtype = N.float) # temporary array
    for (x, y, w) in zip(X, Y, W): # sum over all vectors to be aligned
        xy = N.subtract(x, y, xy)
        M[0,0] += N.dot(xy, xy) * w
        # the sense of rotation is changed referring to the citation above.
        # i.e using 'N.cross(y, x) # instead of 'N.cross(x, y)'.
        M[1:4,0] += 2*N.cross(y, x) * w
        # the upper triangular part of the matrix is not considered if
        # using 'N.linalg.eigh' for solving the eigenvalue problem,
        # uncomment the following line to create the full matrix:
        # M[0,1:4] += 2*N.cross(y, x) * w
        xy = N.add(x, y, xy)
        xy.fill(N.dot(xy, xy))
        M[1:4,1:4] += ( N.diag(xy) - 2*(N.outer(x, y) + N.outer(y, x)) ) * w
    # only use the first eigenvector, which refers to the smallest
    # Eigenvalue
    (eigVal, eigVec) = N.linalg.eigh(M)
    eigV = zip(eigVal, eigVec.T)
    eigV.sort(key = itemgetter(0))
    q = Quaternion(eigV[0][1])
    # make largest element positive
    if q.q[N.abs(q.q).argmax()] < 0: q.q *= -1

    # rotate X (which is shifted to it's center of rotation) and shift
    # to the center of rotation of Y.
    X = (q.rotate(X.T)).T + yC
    # calculate the root mean square deviation
    if RMSD:
        RMSD = N.sqrt(sum( N.dot(x-y, x-y) for (x,y) in zip(X, Y+yC))/n)
    else:
        RMSD = None

    # return the RMSD and a tuple of the first shift vector, the rotation
    # quaternion and the second shift vector.
    return (X, RMSD, (-xC, q,  yC))
