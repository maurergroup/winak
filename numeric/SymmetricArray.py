
from thctk.numeric import *
import thctk.numeric.cSymmetricArrayTools as C
from thctk.numeric.cSymmetricArrayTools import Cache2D

def SymmetricArray2D(n, array = None, dtype = N.float):
    if array is None: 
        array = N.empty(n*(n+1)/2, dtype)
    else: 
        array = N.asarray(array)
        assert array.shape == (n*(n+1)/2,)

    if array.dtype == N.float:
        return array.view(C.dSymmetricArray2D)
    elif array.dtype == N.int:
        return array.view(C.iSymmetricArray2D)
    else:
        return array.view(C.SymmetricArray2D)

def Cache1D(nMax, cache = None):
    n = nMax
    if cache is None:
        cache =  N.empty(n*(n+1)/2, dtype = N.float).view(C.Cache1D)
    else:
        assert cache.shape == (n*(n+1)/2,)
        assert cache.dtype == N.float
        cache = cache.view(C.Cache1D)
    cache.nMax = nMax
    return cache

def Cache2D(nMax, cache = None):
    n = nMax*nMax
    if cache is None:
        cache =  N.empty(n*(n+1)/2, dtype = N.float).view(C.Cache2D)
    else:
        assert cache.shape == (n*(n+1)/2,)
        assert cache.dtype == N.float
        cache = cache.view(C.Cache2D)
    cache.nMax = nMax
    return cache

def Cache3D(nMax, cache = None):
    n = nMax*nMax*nMax
    if cache is None:
        cache =  N.empty(n*(n+1)/2, dtype = N.float).view(C.Cache3D)
    else:
        assert cache.shape == (n*(n+1)/2,)
        assert cache.dtype == N.float
        cache = cache.view(C.Cache3D)
    cache.nMax = nMax
    return cache

def Cache4D(nMax, cache = None):
    n = nMax*nMax
    n *= n
    if cache is None:
        cache =  N.empty(n*(n+1)/2, dtype = N.float).view(C.Cache4D)
    else:
        assert cache.shape == (n*(n+1)/2,)
        assert cache.dtype == N.float
        cache = cache.view(C.Cache3D)
    cache.nMax = nMax
    return cache

