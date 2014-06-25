      subroutine indices(ind1, ind2, length, i, j, n, m, tmpi, tmpj)
          integer k, ind1, ind2, tmpi(length), tmpj(length)
          integer i(length), j(length), n(length), m(length)
            
          tmpi(length) = ind1
          tmpj(length) = ind2
          i(length) = MOD(ind1, n(length))
          j(length) = MOD(ind2, m(length))

          do 10 k = length-1, 1, -1
                tmpi(k) = (tmpi(k+1) - i(k+1)) / n(k+1)
                tmpj(k) = (tmpj(k+1) - j(k+1)) / m(k+1)
                i(k) = MOD(tmpi(k), n(k))
                j(k) = MOD(tmpj(k), m(k))
  10      continue
      end
