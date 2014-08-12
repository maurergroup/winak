import numpy as np

class Vector:
    """ 
    Stub of a Vector class, similar to the one found in Scientific.Geometry.
    Only functionality, that is needed in InternalCoordinates.py, is
    implemented.
    """

    is_vec=True

    def __init__(self,x):
        self.components=np.array(x)
    
    def __sub__(self,o):
        return Vector(self.components-o.components)

    def __rsub__(self,o):
        return Vector(o.components-self.components)

    def __mul__(self,o):
        ret=None
        if(hasattr(o,'is_vec')):
            ret=np.sum(self.components*o.components)
        else:
            ret=Vector(o*self.components)
        return ret

    def __add__(self, o):
        return Vector(self.components+o.components)

    __radd__ = __add__

    def __rmul__(self,o):
       return Vector(o*self.components)
   
    def length(self):
        return np.sqrt(np.sum(self.components*self.components))

    def normal(self):
        l=self.length()
        if l==0:
            raise ZeroDivisionError('Cannot normalize a zero-length vector, silly')
        return Vector(self.components/l)

    def cross(self,o):
        if not hasattr(o,'is_vec'):
            raise TypeError('Cross products only possible between vectors')
        return Vector([self.components[1]*o.components[2]-self.components[2]*o.components[1],self.components[2]*o.components[0]-self.components[0]*o.components[2],self.components[0]*o.components[1]-self.components[1]*o.components[0]])
            
