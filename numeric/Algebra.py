# thctk.numeric.Algebra
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2002 Christoph Scheurer
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
    This Module provides functions for handling operator algebra
"""

import copy
from types import IntType, LongType, FloatType, ComplexType

def numericType(a):
    ta = type(a)
    return (ta == IntType or ta == LongType or ta == FloatType or
            ta == ComplexType)

class Element:

    def __init__(self, name = 'Element', scalar = 1):
        self.scalar = scalar
        self.terms = [self,]
        self.name = name
        self.type = 0

    def __nonzero__(self):
        return self.scalar != 0

    def __repr__(self):
        return self.reprScalar(self.name)

    def reprScalar(self,s):
        if self.scalar == 1: return s
        elif self.scalar == 0: return '0'
        elif self.scalar == -1: return '(-' + s + ')'
        else: return '(' + `self.scalar` + '*' + s + ')'

    def copy(self, name = ''):
        other = copy.deepcopy(self)
        other.terms = self.terms
        if name: other.name = name
        return other

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, key):
        return self.terms[key]

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        other = self.copy()
        other.scalar = - self.scalar
        return other

    def __add__(self, other):
        return Sum(self, other)

    def __sub__(self, other):
        return Sum(self, - other)

    def __mul__(self, other):
        return Product(self, other)

    def __div__(self, other):
        if numericType(other): return Product(self, 1./other)
        else: raise(TypeError, 'unsupported operand type(s) for /')

    def __radd__(self, other):
        return Sum(self, other)

    def __rsub__(self, other):
        return Sum(- self, other)

    def __rmul__(self, other):
        return Product(self, other)

    def __iadd__(self, other):
        return Sum(self, other)

    def __isub__(self, other):
        return Sum(self, - other)

    def __imul__(self, other):
        return Product(self, other)

Id = Element('I')
Id.isId = 1

class Operation(Element):

    def __init__(self, a, b,
                 commutative = 1, symbol = '', product = 0, sum = 0):
        self.terms = []
        self.commutative = commutative
        self.symbol = symbol
        self.scalar = 1
        if product * sum != 0:
            raise(TypeError, 'Operation has to be Sum or Product')
        if sum:
            self.issum = 0
            self.type = 1
            if not symbol: symbol = '+'
        elif product:
            self.isproduct = 0
            self.type = 2
            if not symbol: symbol = '*'
        else: raise(TypeError, 'Operation has to be Sum or Product')
        self.symbol = symbol

        if product:
            if (numericType(a) or hasattr(a, 'isId')) and \
            (numericType(b) or hasattr(b, 'isId')):
                if hasattr(a, 'scalar') and hasattr(b, 'scalar'):
                    self.scalar = a.scalar*b.scalar
                elif hasattr(a, 'scalar'): self.scalar = a.scalar*b
                elif hasattr(b, 'scalar'): self.scalar = b.scalar*a
                else: self.scalar = a*b
                self.terms = Id.terms
                self.isId = 1
            elif (numericType(a) or hasattr(a, 'isId')) and \
            hasattr(b, 'scalar'):
                if hasattr(a, 'scalar'):
                    self.scalar = a.scalar * b.scalar
                else: self.scalar = a * b.scalar
                self.terms = b.terms
            elif (numericType(b) or hasattr(b, 'isId')) and \
            hasattr(a, 'scalar'):
                if hasattr(b, 'scalar'):
                    self.scalar = a.scalar * b.scalar
                else: self.scalar = b * a.scalar
                self.terms = a.terms
            elif self.assoc(a, b):
                self.terms = a.terms + b.terms
                self.scalar = a.scalar * b.scalar
            else: self.terms = [a, b]
        elif sum:
            if numericType(a) and hasattr(b, 'scalar'):
                aId = Id.copy()
                aId.scalar = a
                self.terms = [aId, b]
            elif numericType(b) and hasattr(a, 'scalar'):
                bId = Id.copy()
                bId.scalar = b
                self.terms = [a, bId]
            elif numericType(a) and numericType(b):
                ab = Id.copy()
                ab.scalar = a+b
                self.terms = [ab,]
            elif a.terms == b.terms:
                self.terms = a.terms
                self.scalar = a.scalar + b.scalar
            elif self.assoc(a, b): self.terms = a.terms + b.terms
            else: self.terms = [a, b]

    def __repr__(self):
        if len(self.terms) == 1: s = `self.terms[0]`
        else:
            s = '('
            for t in self.terms[:-1]: s += `t` + self.symbol
            s += `self.terms[-1]` + ')'
        return self.reprScalar(s)

    def associativity(self, a, b):
        if ( a.type == 0 and b.type == 0 ) or \
           ( a.type == self.type and b.type == 0 ) or \
           ( b.type == self.type and a.type == 0 ) or \
           ( a.type == self.type and b.type == self.type ):
            return 1
        else: return 0

    assoc = associativity


class Sum(Operation):

    def __init__(self, a, b, commutative = 1):
        Operation.__init__(self, a, b, commutative, sum = 1)

class Product(Operation):

    def __init__(self, a, b, commutative = 0):
        Operation.__init__(self, a, b, commutative, product = 1)

if __name__ == '__main__':
    a = Element('a')
    b = Element('b')
    c = Element('c')
    m = Product(a,b)
