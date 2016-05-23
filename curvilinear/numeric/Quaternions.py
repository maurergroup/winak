#
# This file is Copyright 2010 Daniel Strobusch
#
"""
Author: Daniel Strobusch
Date: 14. March 2010

This package contains a simple class for describing rotations using quaternions.
Hopefully this will be implemented in NumPy, SciPy or another standard library
soon.
"""

from winak.curvilinear.numeric import *
TRANSPOSER = N.array([1., -1., -1., -1.])

class TransposeDescriptor(object):
    def __get__(self, instance, owner = None):
        return instance.q * TRANSPOSER
    def __set__(self, instance, value):
        if not isinstance(instance, Quaternion):
            raise TypeError(
                "'%s' is not an instance of class 'Quaternion'." %str(instance))
        instance.q = Quaternion(value.q * TRANSPOSER)

class InverseDescriptor(object):
    def __get__(self, instance, owner = None):
        return  Quaternion(instance.T *1./N.dot(instance.q, instance.q))

class Quaternion(object):
    __slots__ = ('q')

    T = TransposeDescriptor()
    I = InverseDescriptor()

    def __init__(self, arg):
        """
        Initialize a quaternion either by angle and axis or by an array of
        length 4.
        """
        # create from pair of (angle, axis)
        if len(arg) == 2 and isinstance(arg[0] ,float) and len(arg[1]) == 3:
            a = arg[0]/2. # divide angle by 2
            axis = N.array(arg[1], dtype = N.float64) # set axis
            axis /= N.sqrt(N.dot(axis, axis)) # norm axis
            axis *= N.sin(a) # multiply with sin(angle/2)
            self.q = N.array([N.cos(a)], dtype = N.float64) # first part of 'q'
            self.q = N.concatenate((self.q, axis)) # create whole quaternion
        elif len(arg) == 4 and all(isinstance(i, (float, int)) for i in arg):
            self.q = N.array(arg, dtype = N.float64)
        else:
            raise TypeError("Can't construct a quaternion from given " +
                    "arguments '%s'" %str(arg))

    def norm(self):
        return N.sqrt(N.dot(self.q, self.q))

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError(
                "'%s' is not an instance of class 'Quaternion'." %str(other))
        q = self.q
        p = other.q
        return Quaternion(q + p)

    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError(
                "'%s' is not an instance of class 'Quaternion'." %str(other))
        q = self.q
        p = other.q
        return Quaternion(q - p)

    def __mul__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError(
                "'%s' is not an instance of class 'Quaternion'." %str(other))
        q = self.q
        p = other.q
        qp = N.array(N.empty(4))
        qp[0] = q[0]*p[0] - N.dot(q[1:], p[1:])
        qp[1:] = N.cross(q[1:], p[1:]) + q[0]*p[1:] + p[0]*q[1:]
        return Quaternion(qp)

    def __div__(self, other):
        q = self.q
        if not isinstance( other, (int, float)):
            raise TypeError("Division is only defined for scalars.")
        return Quaternion(q/other)

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError(
                "'%s' is not an instance of class 'Quaternion'." %str(other))
        p = other.q
        if (self.q == p ).all():
            return True
        else:
            return False

    def __str__(self):
        return "Quaternion(%s)" %self.q

    def __repr__(self):
        return "Quaternion(%s at %s)" %(self.q, hex(id(self)) )

    def rotate(self, v):
        """
        Return a vector or matrix rotated by quaternion.
        """
        assert v.shape[0] == 3, "Given vector or vectors are of wrong shape."
        q = self.q
        return N.array([
        2*(v[2]*q[0]*q[2] + v[1]*q[1]*q[2] - v[1]*q[0]*q[3] +
           v[2]*q[1]*q[3]) + v[0]*
         (q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3])
        ,
        2*(-(v[2]*q[0]*q[1]) + v[0]*q[1]*q[2] + v[0]*q[0]*q[3] +
           v[2]*q[2]*q[3]) + v[1]*
         (q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3])
        ,
        2*(v[1]*q[0]*q[1] - v[0]*q[0]*q[2] + v[0]*q[1]*q[3] +
           v[1]*q[2]*q[3]) + v[2]*
         (q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3])
        ])

    def getAngle(self):
        """
        Get the angle of return described by the quaternion.
        """
        q = self.q
        return 2*N.arccos(q[0])

    def getAxis(self):
        """
        Get the axis of rotation described by the quaternion.
        """
        q = self.q
        return q[1:] / N.sin(self.getAngle()/2.)

    def getRotationMatrix(self):
        """
        Return a matrix describing the rotation.
        """
        q = self.q
        return N.array([
            [ q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3] , 2*(-q[0]*q[3]+q[1]*q[2])               , 2*(q[0]*q[2]+q[1]*q[3])                ],
            [2*(q[0]*q[3]+q[1]*q[2])                  , q[0]*q[0]+q[2]*q[2]-q[1]*q[1]-q[3]*q[3], 2*(-q[0]*q[1]+q[2]*q[3])               ],
            [2*(-q[0]*q[2]+q[1]*q[3])                 , 2*(q[0]*q[1]+q[2]*q[3])                , q[0]*q[0]+q[3]*q[3]-q[1]*q[1]-q[2]*q[2]]
            ])

    def getTaitBryanZYXAngles(self):
        """
        Calculates the three angles phi, theta, psi for
        consecutive rotation around XYZ
        The rotation order is first x, than y, than z axis
        """

        q = self.q

        w2 = q[0]*q[0]
        x2 = q[1]*q[1]
        y2 = q[2]*q[2]
        z2 = q[3]*q[3]
        L = w2 + x2 + y2 + z2
        #zxy convention
        #abcd = q[0]*q[1] + q[2]*q[3]
        #eps = 1.e-7
        #if (abcd > (0.5-eps)*L):
            #yaw = 2. * N.arctan2(q[2]*q[0])
            #pitch = N.pi
            #roll = 0.0
        #elif (abcd < (-0.5+eps)*L):
            #yaw = -2. * N.arctan2(q[2]*q[0])
            #pitch = -N.pi
            #roll = 0.0
        #else:
            #adbc = q[0]*q[3] - q[1]*q[2]
            #acbd = q[0]*q[2] - q[1]*q[3]
            #yaw = N.arctan2(2.*adbc, 1. - 2.*(z2+x2))
            #pitch = N.arcsin(2.*abcd/L)
            #roll = N.arctan2(2.*acbd, 1.-2.*(y2+x2))

        #return pitch, yaw, roll

#xyz convention
        acbd = q[0]*q[2] + q[1]*q[3]
        eps = 1.e-8
        pitch = N.arcsin(2.*acbd/L);
        if (acbd > (0.5-eps)*L):
            yaw = N.arctan2(q[1], q[0]);
            pitch = N.pi/2.;
            roll = 0;
        elif (acbd < (-0.5+eps)*L):
            yaw = -N.arctan2(q[1], q[0]);
            pitch = -N.pi/2.;
            roll = 0;
        else:
            abcd = q[0]*q[1] - q[2]*q[3]
            adbc = q[0]*q[3] - q[1]*q[2]
            yaw = N.arctan2(2.*abcd, 1. - 2.*(y2+x2));
            pitch = N.arcsin(2.*acbd/L);
            roll = N.arctan2(2.*adbc, 1. - 2.*(y2+z2));

        return yaw, pitch, roll

        #phi = N.arctan2(-R[1,2],R[2,2])
        #theta = N.arcsin(-R[0,2])
        #psi = N.arctan2(-R[0,1],R[0,0])
        #return phi, theta, psi

