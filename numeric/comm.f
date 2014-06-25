c----------------------------------------------------------------------c
c Commutator routines
c----------------------------------------------------------------------c
c description:
c------------
c routines to compute [mu,rho] commutator in vector format:
c
c [kron(1,mu) - kron(mu,1)] * vec(rho),
c
c using fblas ?axpy functions. mu is a sparse matrix in csr-format 
c and rho is a dense matrix (vectorized).
c
c comrr - mu and rho are real
c comzz - mu and rho are complex
c comrz - mu is real and rho is complex
c 
c-----------------------------------------------------------------------
c on entry:
c ---------
c n  = integer. Dimension of the matrix
c nnz = integer. Number of nonzeros
c
c ci,
c cj,
c cx = Dipol-Matrix MU in compressed sparse row format.
c
c rho = Density matrix in vector format - vec(rho)
c
c on entry/return:
c----------------
c rho2 = Vector of same size and type as rho. Commutator of mu and
c   will be saved in this vector
c
c----------------------------------------------------------------------- 
c----------------------------------------------------------------------c
      subroutine comrr (n,nnz,ci,cj,cx,rho,rho2,nb,ne)
      implicit none
      double precision cx(*), alpha, rho(*), rho2(*)
      integer ci(*), cj(*), n, nnz, nb, ne, i, j
      integer colval, offsetJ, offsetI1, offsetI2, stride1, strideN
      stride1 = 1
      strideN = n

      do 100 i=1, n*n
        rho2(i) = 0.d0
 100   enddo

      do 110 i=nb, ne
        offsetI1 = i
        offsetI2 = (i-1) * n + 1
        do 210 j=ci(i), ci(i+1)-1
          colval = cj(j)
          alpha  = cx(j)
          offsetJ = colval
          call daxpy(n, -alpha, rho(offsetJ), strideN, rho2(offsetI1), 
     *       strideN)
          offsetJ = (colval-1) * n + 1
          call daxpy(n, alpha, rho(offsetJ), stride1, rho2(offsetI2), 
     *       stride1)
 210  continue
 110  enddo
      return
      end
c----------------------------------------------------------------------c
c----------------------------------------------------------------------c
      subroutine comzz (n,nnz,ci,cj,cx,rho,rho2,nb,ne)
      implicit none
      double complex cx(*), alpha, rho(*), rho2(*)
      integer ci(*), cj(*), n, nnz, nb, ne, i, j
      integer colval, offsetJ, offsetI1, offsetI2, stride1, strideN
      stride1 = 1
      strideN = n

      do 100 i=1, n*n
        rho2(i) = (0.d0, 0.d0)
 100  enddo

      do 110 i=nb, ne
        offsetI1 = i
        offsetI2 = (i-1) * n + 1
        do 210 j=ci(i), ci(i+1)-1
          colval = cj(j)
          alpha  = cx(j)
          offsetJ = colval
          call zaxpy(n, -alpha, rho(offsetJ), strideN, rho2(offsetI1), 
     *       strideN)
          offsetJ = (colval-1) * n + 1
          call zaxpy(n, alpha, rho(offsetJ), stride1, rho2(offsetI2), 
     *       stride1)
 210  continue
 110  enddo
      return
      end
c----------------------------------------------------------------------c
c----------------------------------------------------------------------c
      subroutine comrz (n,nnz,ci,cj,cx,rho,rho2,nb,ne)
      implicit none
      double precision cx(*), alpha, rho(*), rho2(*)
      integer ci(*), cj(*), n, nnz, nb, ne, i, j
      integer colval, offsetJ, offsetI1, offsetI2, offsetI3, offsetI4
      integer stride1, strideN
      stride1 = 2
      strideN = 2*n

      do 100 i=1, 2*n*n
        rho2(i) = 0.d0
 100  enddo

      do 110 i=nb, ne
        offsetI1 = 2*i-1
        offsetI2 = (i-1) * 2 * n + 1
        offsetI3 = offsetI1 + 1
        offsetI4 = offsetI2 + 1
        do 210 j=ci(i), ci(i+1)-1
          colval = cj(j)
          alpha  = cx(j)

          offsetJ = 2*colval-1
          call daxpy(n, -alpha, rho(offsetJ), strideN, 
     *       rho2(offsetI1), strideN)
          offsetJ = offsetJ + 1
          call daxpy(n, -alpha, rho(offsetJ), strideN, rho2(offsetI3), 
     *       strideN)
          offsetJ = (colval-1) * 2 * n + 1
          call daxpy(n, alpha, rho(offsetJ), stride1, rho2(offsetI2),
     *       stride1)
          offsetJ = offsetJ + 1
          call daxpy(n, alpha, rho(offsetJ), stride1, rho2(offsetI4),
     *       stride1)
 210  continue
 110  enddo
      return
      end
c----------------------------------------------------------------------c
c----------------------------------------------------------------------c
