# thctk.numeric.CGO
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
CGO - Constrained Global Optimization
"""

import copy, random



class GM:

    """
    Combination of Glauber spin-flip and Metropolis

    Eric Lewin Altschuler, Timothy J. Williams, Edward R. Ratner, Farid Dowla,
    and Frederick Wooten,  Phys. Rev. Lett. 72, 2671-2674 (1994)
    """

    def __init__(self, x0 = None, f = None, schedule = None,
                 choose = None, rand = random.random):
        self.setF(f)
        self.setSchedule(schedule)
        self.setChoose(choose)
        self.setX0(x0)
        self.rand = rand
        self.init()

    def __len__(self): return len(self.x0)

    def init(self):
        self.step = 0
        self.beta = 0.
        self.C = 0.
        try:
            self.schedule()
        except:
            pass
        try:
            self.fx, self.gx = self.f(self.x0)
        except:
            pass
        self.x = copy.copy(self.x0)

    reset = init

    def setF(self, f):
        self.f = f

    def setSchedule(self, schedule):
# The schedule function has to set self.beta and self.C at every step
# The optimizer will stop when self.schedule() returns 0
        self.schedule = schedule

    def setChoose(self, choose):
        self.choose = choose

    def setX0(self, x0):
        self.x0 = x0

    def __call__(self):
        while self.schedule():
            x = []
            for i in range(len(self.x0)):
                if self.rand() < 1/(1 + math.exp(- self.beta * \
                    (self.gx[i] - self.C))):
                    x.append(self.choose())
                else:
                    x.append(self.x[i])
            fx, gx = self.f(x)
            if fx < self.fx or \
                self.rand() < math.exp(- self.beta * (fx - self.fx)):
                self.gx = gx
                self.fx = fx
                self.x = x
