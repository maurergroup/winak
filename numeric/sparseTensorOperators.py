from thctk.numeric import *
from thctk.numeric.tensorIndices import indices as tensorIndexSearchFortran
from thctk.numeric._numeric import tensorOperatorIndexPair

class TensorOperator:
    def __init__(self, operators=[], default="C"):
        self.operators = operators
        lop = len(operators) 
        self.lop = lop
        self.ti, self.tj = N.zeros(lop, nxInt), N.zeros(lop, nxInt)
        self.i = N.zeros(lop, nxInt)
        self.j = N.zeros(lop, nxInt)
        self.n = N.array([operators[i].shape[0] for i in range(lop)], nxInt)
        self.m = N.array([operators[i].shape[1] for i in range(lop)], nxInt)
        if default=="C":
            self.tensorIndexSearch = self.tensorIndexSearchC
        elif default=="Fortran":
            self.tensorIndexSearch = tensorIndexSearchFortran
        elif default=="Python":
            self.tensorIndexSearch = self.tensorIndexSearchPython
        else:
            raise NotImplementedError, 'default must be either "C" or "Fortran" or "Python".\ncurrent default value is "%s"'%default
        self.kron()
        self.t = self.tensor

    def kron(self):
        tensor = 1
        for i in self.operators:
            tensor = N.kron(tensor, i)
        self.tensor = tensor

    def searchIndexPair(self, I, J, debug=False):
        self.tensorIndexSearch(I, J, self.lop, self.i, self.j, self.n, self.m, self.ti, self.tj)

    def compValueArray(self, indexArray):
        """ indexArray must be an integer array with shape (n,2)
            where n stays for number of index pairs
        """
        if indexArray.shape[1] != 2:
            raise NotImplementedError, "shape of indexArray must be (n,2)"
        valuesArray = N.ones(indexArray.shape[0], nxFloat)
        for a in range(len(valuesArray)):
            k,l = indexArray[a]
            self.searchIndexPair(k,l)
            for b in range(self.lop):
                valuesArray[a] *= self.operators[b][self.i[b], self.j[b]]
        return valuesArray

    def tensorIndexSearchC(self, I, J, lop, i, j, n, m, tmpi, tmpj, debug=False):
        tensorOperatorIndexPair((I,J), (i,j), (n,m))

    def tensorIndexSearchPython(self, I, J, lop, i, j, n, m, tmpi, tmpj, debug=False):
        for k in range(lop-1,-1,-1):
            try:
                tmpi[k] = (tmpi[k+1] - i[k+1]) / n[k+1]
                tmpj[k] = (tmpj[k+1] - j[k+1]) / m[k+1]
            except IndexError:
                tmpi[k] = I
                tmpj[k] = J
            i[k] = tmpi[k] % n[k]
            j[k] = tmpj[k] % m[k]
        if debug:
            st = "%i %i\n"%(I,J)
            for k in range(lop-1,-1,-1):
                st+= "%i, %i, %i, %i, %i, %i\n"%(k, i[k],j[k],n[k],tmpi[k],tmpj[k])
            print st

