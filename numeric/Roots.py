# thctk.numeric.Roots
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
Finding Roots f(x) = 0 of functions.
"""

class Roots:

    def __init__(self, f, df = None):
        self.f = f
        self.df = df
        self.points = []

    def BracketOut(self, a, b, fa = None, fb = None, fac = 1.6, ntry = 50):
        """
        Numerical Receipes (2nd ed.) ch. 9.1
        """
        if a == b: raise ValueError
        if fa is None:
            fa = self.f(a)
            self.points.append((a, fa))
        if fb is None:
            fb = self.f(b)
            self.points.append((b, fb))
        n = 0
        while n < ntry:
            if fa*fb < 0: return (a, fa), (b, fb)
            if abs(fa) < abs(fb):
                a += fac * (a - b)
                fa = self.f(a)
                self.points.append((a, fa))
            else:
                b += fac * (b - a)
                fb = self.f(b)
                self.points.append((b, fb))
            n += 1
        return None

    def BracketInterior(self, a, b, n):
        """
        Numerical Receipes (2nd ed.) ch. 9.1
        """
        r = []
        x = a
        dx = (b - a)/float(n)
        fp = self.f(x)
        self.points.append((x, fp))
        for i in range(1, n+1):
            x += dx
            fc = self.f(x)
            self.points.append((x, fc))
            if fp*fc < 0: r.append(((x-dx, fp), (x, fc)))
            fp = fc
        return r

    def Bisect(self, a, b, xacc = None, nmax = 40):
        """
        Numerical Receipes (2nd ed.) ch. 9.1
        """
        if xacc == None: xacc = 1e-10 * (abs(a) + abs(b))/2.
        fmid = self.f(b)
        self.points.append((b, fmid))
        f = self.f(a)
        self.points.append((a, f))
        if f*fmid >= 0: raise ValueError
        if f < 0:
            x0 = a
            dx = b - a
        else:
            x0 = b
            dx = a - b
        n = 0
        while n < nmax:
            dx *= 0.5
            xmid = x0 + dx
            fmid = self.f(xmid)
            self.points.append((xmid, fmid))
            if fmid <= 0: x0 = xmid
            if abs(dx) < xacc or fmid == 0: return x0
            n += 1
        return None

    def Brent(self, a, b, tol, fa = None, fb = None, nmax=100, eps=3.0e-8):
        """
        Numerical Receipes (2nd ed.) ch. 9.3
        """
        if fa is None:
            fa = self.f(a)
            self.points.append((a, fa))
        if fb is None:
            fb = self.f(b)
            self.points.append((b, fb))
        if fa*fb > 0: raise ValueError
        c = b
        fc = fb
        n = 0
        while n < nmax:
            n += 1
            if fb*fc > 0:
                c = a; fc = fa
                d = b -a
                e = d
            if abs(fc) < abs(fb):
                a = b; fa = fb
                b = c; fb = fc
                c = a; fc = fa
            tol1 = 2*eps*abs(b) + tol/2. # convergence check
            xm = (c - b)/2.
            if abs(xm) < tol1 or fb == 0: return b
            if abs(e) >= tol1 and abs(fa) > abs(fb):
                s = fb/fa   # attempt inverse quadratic interpolation
                if a == c:
                    p = 2*xm*s
                    q = 1 - s
                else:
                    q = fa/fc
                    r = fb/fc
                    p = s*(2*xm*q*(q-r) - (b-a)*(r-1))
                    q = (q-1)*(r-1)*(s-1)
                if p > 0: q = -q
                p = abs(p)
                if 2*p < min(3*xm*q - abs(tol1*q), abs(e*q)):
                    e = d   # accept interpolation
                    d = p/q
                else:
                    d = xm  # interpolation failed, use bisection
                    e = d
            else:           # bounds decreasing too slowly, use bisection
                d = xm
                e = d
            a = b; fa = fb
            if abs(d) > tol1: b += d
            else: b += abs(tol1) * xm/abs(xm) # Fortran sign(tol1, xm)
            fb = self.f(b)
            self.points.append((b, fb))
        return None

    def Newton(self, a, b, xacc = None, fa = None, fb = None, h = 1.0e-5, nmax=100):
        """
        Using a combination of NR and bisection find the root of a
        function between a and b.
        Numerical Receipes (2nd ed.) ch. 9.3
        f'(x) = f(x+h)-f(x) / h
        """
        if self.df is None:
            df = lambda x: (self.f(x + h) - fxm) / h
        else:
            df = self.df
        if xacc is None : xacc = 1e-6
        if fa is None:
            fa = self.f(a)
        if fb is None:
            fb = self.f(b)
        if fa*fb > 0: raise ValueError
        if fa < 0:
            xl = a
            xh = b
        else:
            xl = b
            xh = a
        xm = 0.5*(a + b)
        dxold = abs( b - a)
        dx = dxold
        fxm = self.f(xm)
        fprixm = df(xm)
        n = 0
        while n < nmax:
            n += 1
            if (((xm-xh)*fprixm-fxm)*((xm-xl)*fprixm-fxm) > 0.) \
                or (abs(2.*fxm) > abs(dxold*fprixm)):
                dxold = dx
                dx = 0.5*(xh-xl)
                xm = xl + dx
                if xl == xm : return xm
            else:
                dxold = dx
                dx = fxm/fprixm
                temp = xm
                xm -= dx
                if temp == xm : return xm
            if abs(dx) < xacc: return xm
            fxm = self.f(xm)
            fprixm = df(xm)
            if fxm < 0:
                xl = xm
            else:
                xh = xm
        return None
