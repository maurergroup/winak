c-----------------------------------------------------------------------
      subroutine csrmsr (n,a,ja,ia,ao,jao,wk,iwk)
      real*8 a(*),ao(*),wk(n)
      integer ia(n+1),ja(*),jao(*),iwk(n+1)
c----------------------------------------------------------------------- 
c Compressed Sparse Row   to      Modified - Sparse Row 
c                                 Sparse row with separate main diagonal
c----------------------------------------------------------------------- 
c converts a general sparse matrix a, ja, ia into 
c a compressed matrix using a separated diagonal (referred to as
c the bell-labs format as it is used by bell labs semi conductor
c group. We refer to it here as the modified sparse row format.
c Note: this has been coded in such a way that one can overwrite
c the output matrix onto the input matrix if desired by a call of
c the form 
c
c     call csrmsr (n, a, ja, ia, a, ja, wk,iwk)
c
c In case ao, jao, are different from a, ja, then one can
c use ao, jao as the work arrays in the calling sequence:
c
c     call csrmsr (n, a, ja, ia, ao, jao, ao,jao)
c
c----------------------------------------------------------------------- 
c
c on entry :
c---------
c a, ja, ia = matrix in csr format. note that the 
c	     algorithm is in place: ao, jao can be the same
c            as a, ja, in which case it will be overwritten on it
c            upon return.
c	 
c on return :
c-----------
c
c ao, jao  = sparse matrix in modified sparse row storage format:
c	   +  ao(1:n) contains the diagonal of the matrix. 
c	   +  ao(n+2:nnz) contains the nondiagonal elements of the
c             matrix, stored rowwise.
c	   +  jao(n+2:nnz) : their column indices
c	   +  jao(1:n+1) contains the pointer array for the nondiagonal
c             elements in ao(n+1:nnz) and jao(n+2:nnz).
c             i.e., for i .le. n+1 jao(i) points to beginning of row i 
c	      in arrays ao, jao.
c	       here nnz = number of nonzero elements+1 
c work arrays:
c------------
c wk	= real work array of length n
c iwk   = integer work array of length n+1
c
c notes: 
c------- 
c        Algorithm is in place.  i.e. both:
c
c          call csrmsr (n, a, ja, ia, ao, jao, ao,jao)
c          (in which  ao, jao, are different from a, ja)
c           and
c          call csrmsr (n, a, ja, ia, a, ja, wk,iwk) 
c          (in which  wk, jwk, are different from a, ja)
c        are OK.
c--------
c coded by Y. Saad Sep. 1989. Rechecked Feb 27, 1990.
c-----------------------------------------------------------------------
      icount = 0
c
c store away diagonal elements and count nonzero diagonal elements.
c
      do 1 i=1,n
         wk(i) = 0.0d0
         iwk(i+1) = ia(i+1)-ia(i)
         do 2 k=ia(i),ia(i+1)-1
            if (ja(k) .eq. i) then
               wk(i) = a(k)
               icount = icount + 1 
               iwk(i+1) = iwk(i+1)-1
            endif
 2       continue
 1    continue
c     
c compute total length
c     
      iptr = n + ia(n+1) - icount
c     
c     copy backwards (to avoid collisions)
c     
      do 500 ii=n,1,-1
         do 100 k=ia(ii+1)-1,ia(ii),-1
            j = ja(k)
            if (j .ne. ii) then
c              write(*,*), ii, k, j, a(k)
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr-1
            endif
 100     continue
 500  continue
c
c compute pointer values and copy wk(*)
c
c     jao(1) = n+2
      jao(1) = 1
      do 600 i=1,n
         ao(i) = wk(i) 
         jao(i+1) = jao(i)+iwk(i+1)
 600  continue
      return	
c------------ end of subroutine csrmsr ---------------------------------
c----------------------------------------------------------------------- 
      end
c-----------------------------------------------------------------------
      subroutine csrmsrcomplex (n,a,ja,ia,ao,jao,wk,iwk)
      complex*16 a(*),ao(*),wk(n)
      integer ia(n+1),ja(*),jao(*),iwk(n+1)
c----------------------------------------------------------------------- 
c Compressed Sparse Row   to      Modified - Sparse Row 
c                                 Sparse row with separate main diagonal
c----------------------------------------------------------------------- 
c converts a general sparse matrix a, ja, ia into 
c a compressed matrix using a separated diagonal (referred to as
c the bell-labs format as it is used by bell labs semi conductor
c group. We refer to it here as the modified sparse row format.
c Note: this has been coded in such a way that one can overwrite
c the output matrix onto the input matrix if desired by a call of
c the form 
c
c     call csrmsr (n, a, ja, ia, a, ja, wk,iwk)
c
c In case ao, jao, are different from a, ja, then one can
c use ao, jao as the work arrays in the calling sequence:
c
c     call csrmsr (n, a, ja, ia, ao, jao, ao,jao)
c
c----------------------------------------------------------------------- 
c
c on entry :
c---------
c a, ja, ia = matrix in csr format. note that the 
c	     algorithm is in place: ao, jao can be the same
c            as a, ja, in which case it will be overwritten on it
c            upon return.
c	 
c on return :
c-----------
c
c ao, jao  = sparse matrix in modified sparse row storage format:
c	   +  ao(1:n) contains the diagonal of the matrix. 
c	   +  ao(n+2:nnz) contains the nondiagonal elements of the
c             matrix, stored rowwise.
c	   +  jao(n+2:nnz) : their column indices
c	   +  jao(1:n+1) contains the pointer array for the nondiagonal
c             elements in ao(n+1:nnz) and jao(n+2:nnz).
c             i.e., for i .le. n+1 jao(i) points to beginning of row i 
c	      in arrays ao, jao.
c	       here nnz = number of nonzero elements+1 
c work arrays:
c------------
c wk	= real work array of length n
c iwk   = integer work array of length n+1
c
c notes: 
c------- 
c        Algorithm is in place.  i.e. both:
c
c          call csrmsr (n, a, ja, ia, ao, jao, ao,jao)
c          (in which  ao, jao, are different from a, ja)
c           and
c          call csrmsr (n, a, ja, ia, a, ja, wk,iwk) 
c          (in which  wk, jwk, are different from a, ja)
c        are OK.
c--------
c coded by Y. Saad Sep. 1989. Rechecked Feb 27, 1990.
c-----------------------------------------------------------------------
      icount = 0
c
c store away diagonal elements and count nonzero diagonal elements.
c
      do 1 i=1,n
         wk(i) = 0.0d0
         iwk(i+1) = ia(i+1)-ia(i)
         do 2 k=ia(i),ia(i+1)-1
            if (ja(k) .eq. i) then
               wk(i) = a(k)
               icount = icount + 1 
               iwk(i+1) = iwk(i+1)-1
            endif
 2       continue
 1    continue
c     
c compute total length
c     
      iptr = n + ia(n+1) - icount
c     
c     copy backwards (to avoid collisions)
c     
      do 500 ii=n,1,-1
         do 100 k=ia(ii+1)-1,ia(ii),-1
            j = ja(k)
            if (j .ne. ii) then
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr-1
            endif
 100     continue
 500  continue
c
c compute pointer values and copy wk(*)
c
      jao(1) = 1
      do 600 i=1,n
         ao(i) = wk(i) 
         jao(i+1) = jao(i)+iwk(i+1)
 600  continue
      return	
c------------ end of subroutine csrmsr ---------------------------------
c----------------------------------------------------------------------- 
      end
c-----------------------------------------------------------------------
      subroutine msrcsr (n,a,ja,ao,jao,iao,wk,iwk)
      real*8 a(*),ao(*),wk(n)
      integer ja(*),jao(*),iao(n+1),iwk(n+1)
c----------------------------------------------------------------------- 
c       Modified - Sparse Row  to   Compressed Sparse Row   
c
c----------------------------------------------------------------------- 
c converts a compressed matrix using a separated diagonal 
c (modified sparse row format) in the Compressed Sparse Row   
c format.
c does not check for zero elements in the diagonal.
c
c
c on entry :
c---------
c n          = row dimension of matrix
c a, ja      = sparse matrix in msr sparse storage format
c              see routine csrmsr for details on data structure 
c        
c on return :
c-----------
c
c ao,jao,iao = output matrix in csr format.  
c
c work arrays:
c------------
c wk       = real work array of length n
c iwk      = integer work array of length n+1
c
c notes:
c   The original version of this was NOT in place, but has
c   been modified by adding the vector iwk to be in place.
c   The original version had ja instead of iwk everywhere in
c   loop 500.  Modified  Sun 29 May 1994 by R. Bramley (Indiana).
c   
c----------------------------------------------------------------------- 
      logical added
      do 1 i=1,n
         wk(i) = a(i)
         iwk(i) = ja(i)
 1    continue
      iwk(n+1) = ja(n+1)
c     write(*,*) iwk(n), iwk(n+1)
      iao(1) = 1
      iptr = 1
c---------
      do 500 ii=1,n 
         added = .false.
         idiag = iptr + (iwk(ii+1)-iwk(ii)) 
         do 100 k=iwk(ii)+n+1,iwk(ii+1)+n
            j = ja(k)
            if (j .lt. ii) then
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr+1
            elseif (added) then
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr+1
            else 
c add diag element - only reserve a position for it. 
               idiag = iptr
               iptr = iptr+1
               added = .true.
c     then other element
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr+1
            endif
 100     continue
         ao(idiag) = wk(ii)
         jao(idiag) = ii
         if (.not. added) iptr = iptr+1
         iao(ii+1) = iptr 
 500  continue
      return    
c------------ end of subroutine msrcsr --------------------------------- 
c-----------------------------------------------------------------------
      end
c----------------------------------------------------------------------- 
      subroutine msrcsrcomplex (n,a,ja,ao,jao,iao,wk,iwk)
      complex*16 a(*),ao(*),wk(n)
      integer ja(*),jao(*),iao(n+1),iwk(n+1)
c----------------------------------------------------------------------- 
c       Modified - Sparse Row  to   Compressed Sparse Row   
c
c----------------------------------------------------------------------- 
c converts a compressed matrix using a separated diagonal 
c (modified sparse row format) in the Compressed Sparse Row   
c format.
c does not check for zero elements in the diagonal.
c
c
c on entry :
c---------
c n          = row dimension of matrix
c a, ja      = sparse matrix in msr sparse storage format
c              see routine csrmsr for details on data structure 
c        
c on return :
c-----------
c
c ao,jao,iao = output matrix in csr format.  
c
c work arrays:
c------------
c wk       = real work array of length n
c iwk      = integer work array of length n+1
c
c notes:
c   The original version of this was NOT in place, but has
c   been modified by adding the vector iwk to be in place.
c   The original version had ja instead of iwk everywhere in
c   loop 500.  Modified  Sun 29 May 1994 by R. Bramley (Indiana).
c   
c----------------------------------------------------------------------- 
      logical added
      do 1 i=1,n
         wk(i) = a(i)
         iwk(i) = ja(i)
 1    continue
      iwk(n+1) = ja(n+1)
      iao(1) = 1
      iptr = 1
c---------
      do 500 ii=1,n 
         added = .false.
         idiag = iptr + (iwk(ii+1)-iwk(ii)) 
         do 100 k=iwk(ii)+n+1,iwk(ii+1)+n
            j = ja(k)
            if (j .lt. ii) then
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr+1
            elseif (added) then
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr+1
            else 
c add diag element - only reserve a position for it. 
               idiag = iptr
               iptr = iptr+1
               added = .true.
c     then other element
               ao(iptr) = a(k)
               jao(iptr) = j 
               iptr = iptr+1
            endif
 100     continue
         ao(idiag) = wk(ii)
         jao(idiag) = ii
         if (.not. added) iptr = iptr+1
         iao(ii+1) = iptr 
 500  continue
      return    
c------------ end of subroutine msrcsr --------------------------------- 
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine csrcsc (n,m,a,ja,ia,ao,jao,iao)
      integer ia(n+1),iao(m+1),ja(*),jao(*)
      real*8  a(*),ao(*)
c-----------------------------------------------------------------------
c Compressed Sparse Row     to      Compressed Sparse Column
c
c (transposition operation)   Not in place. 
c----------------------------------------------------------------------- 
c Rectangular version.  n is number of rows of CSR matrix,
c                       m (input) is number of columns of CSC matrix.
c----------------------------------------------------------------------- 
c -- not in place --
c this subroutine transposes a matrix stored in a, ja, ia format.
c ---------------
c on entry:
c----------
c n	= number of rows of CSR matrix.
c m    = number of columns of CSC matrix.
c
c a	= real array of length nnz (nnz=number of nonzero elements in input 
c         matrix) containing the nonzero elements.
c ja	= integer array of length nnz containing the column positions
c 	  of the corresponding elements in a.
c ia	= integer of size n+1. ia(k) contains the position in a, ja of
c	  the beginning of the k-th row.
c
c on return:
c ---------- 
c output arguments:
c ao	= real array of size nzz containing the "a" part of the transpose
c jao	= integer array of size nnz containing the column indices.
c iao	= integer array of size n+1 containing the "ia" index array of
c	  the transpose. 
c
c----------------------------------------------------------------------- 
c----------------- compute lengths of rows of transp(A) ----------------
      do 1 i=1,m+1
         iao(i) = 0
 1    continue
      do 3 i=1, n
         do 2 k=ia(i), ia(i+1)-1 
            j = ja(k)+1
            iao(j) = iao(j)+1
 2       continue 
 3    continue
c---------- compute pointers from lengths ------------------------------
      iao(1) = 1 
      do 4 i=1,m
         iao(i+1) = iao(i) + iao(i+1)
 4    continue
c--------------- now do the actual copying ----------------------------- 
      do 6 i=1,n
         do 62 k=ia(i),ia(i+1)-1 
            j = ja(k) 
            next = iao(j)
            ao(next) = a(k)
            jao(next) = i
            iao(j) = next+1
 62      continue
 6    continue
c-------------------------- reshift iao and leave ---------------------- 
      do 7 i=m,1,-1
         iao(i+1) = iao(i)
 7    continue
      iao(1) = 1
c--------------- end of csrcsc  ---------------------------------------- 
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine csrcsccomplex (n,m,a,ja,ia,ao,jao,iao)
      integer ia(n+1),iao(m+1),ja(*),jao(*)
      complex*16  a(*),ao(*)
c-----------------------------------------------------------------------
c Compressed Sparse Row     to      Compressed Sparse Column
c
c (transposition operation)   Not in place. 
c----------------------------------------------------------------------- 
c Rectangular version.  n is number of rows of CSR matrix,
c                       m (input) is number of columns of CSC matrix.
c----------------------------------------------------------------------- 
c -- not in place --
c this subroutine transposes a matrix stored in a, ja, ia format.
c ---------------
c on entry:
c----------
c n	= number of rows of CSR matrix.
c m    = number of columns of CSC matrix.
c
c a	= real array of length nnz (nnz=number of nonzero elements in input 
c         matrix) containing the nonzero elements.
c ja	= integer array of length nnz containing the column positions
c 	  of the corresponding elements in a.
c ia	= integer of size n+1. ia(k) contains the position in a, ja of
c	  the beginning of the k-th row.
c
c on return:
c ---------- 
c output arguments:
c ao	= real array of size nzz containing the "a" part of the transpose
c jao	= integer array of size nnz containing the column indices.
c iao	= integer array of size n+1 containing the "ia" index array of
c	  the transpose. 
c
c----------------------------------------------------------------------- 
c----------------- compute lengths of rows of transp(A) ----------------
      do 1 i=1,m+1
         iao(i) = 0
 1    continue
      do 3 i=1, n
         do 2 k=ia(i), ia(i+1)-1 
            j = ja(k)+1
            iao(j) = iao(j)+1
 2       continue 
 3    continue
c---------- compute pointers from lengths ------------------------------
      iao(1) = 1 
      do 4 i=1,m
         iao(i+1) = iao(i) + iao(i+1)
 4    continue
c--------------- now do the actual copying ----------------------------- 
      do 6 i=1,n
         do 62 k=ia(i),ia(i+1)-1 
            j = ja(k) 
            next = iao(j)
            ao(next) = a(k)
            jao(next) = i
            iao(j) = next+1
 62      continue
 6    continue
c-------------------------- reshift iao and leave ---------------------- 
      do 7 i=m,1,-1
         iao(i+1) = iao(i)
 7    continue
      iao(1) = 1
c--------------- end of csrcsc  ---------------------------------------- 
c-----------------------------------------------------------------------
      end
c-----------------------------------------------------------------------
      subroutine dnscsr(nrow,ncol,nzmax,dns,a,ja,ia,ierr)
      real*8 dns(*),a(*),element
      integer ia(*),ja(*)
c-----------------------------------------------------------------------
c Dense		to    Compressed Row Sparse 
c----------------------------------------------------------------------- 
c
c converts a densely stored matrix into a row orientied
c compactly sparse matrix. ( reverse of csrdns )
c Note: this routine does not check whether an element 
c is small. It considers that a(i,j) is zero if it is exactly
c equal to zero: see test below.
c-----------------------------------------------------------------------
c on entry:
c---------
c
c nrow	= row-dimension of a
c ncol	= column dimension of a
c nzmax = maximum number of nonzero elements allowed. This
c         should be set to be the lengths of the arrays a and ja.
c dns   = input nrow x ncol (dense) matrix.
c
c on return:
c---------- 
c 
c a, ja, ia = value, column, pointer  arrays for output matrix 
c
c ierr	= integer error indicator: 
c         ierr .eq. 0 means normal retur
c         ierr .eq. i means that the the code stopped while
c         processing row number i, because there was no space left in
c         a, and ja (as defined by parameter nzmax).
c----------------------------------------------------------------------- 
      ierr = 0
      next = 1
      ia(1) = 1
      do 4 i=1,nrow
         do 3 j=1, ncol 
            element = dns((i-1)*nrow+j)
            if ((element .eq. 0.0d0) .and. (i .ne. j)) goto 3
c           if (element .eq. 0.0d0) goto 3
            if (next .gt. nzmax) then
               ierr = i
               return
            endif
            ja(next) = j
            a(next) = element
            next = next+1
 3       continue	   
         ia(i+1) = next
 4    continue
      return
c---- end of dnscsr ---------------------------------------------------- 
c----------------------------------------------------------------------- 
      end
c----------------------------------------------------------------------- 
      subroutine dnscsrcomplex(nrow,ncol,nzmax,dns,a,ja,ia,ierr)
      complex*16 dns(*),a(*), element
      integer ia(*),ja(*)
c-----------------------------------------------------------------------
c Dense		to    Compressed Row Sparse 
c----------------------------------------------------------------------- 
c
c converts a densely stored matrix into a row orientied
c compactly sparse matrix. ( reverse of csrdns )
c Note: this routine does not check whether an element 
c is small. It considers that a(i,j) is zero if it is exactly
c equal to zero: see test below.
c-----------------------------------------------------------------------
c on entry:
c---------
c
c nrow	= row-dimension of a
c ncol	= column dimension of a
c nzmax = maximum number of nonzero elements allowed. This
c         should be set to be the lengths of the arrays a and ja.
c dns   = input nrow x ncol (dense) matrix.
c
c on return:
c---------- 
c 
c a, ja, ia = value, column, pointer  arrays for output matrix 
c
c ierr	= integer error indicator: 
c         ierr .eq. 0 means normal retur
c         ierr .eq. i means that the the code stopped while
c         processing row number i, because there was no space left in
c         a, and ja (as defined by parameter nzmax).
c----------------------------------------------------------------------- 
      ierr = 0
      next = 1
      ia(1) = 1
      do 4 i=1,nrow
         do 3 j=1, ncol 
            element = dns((i-1)*nrow+j)
c           if (ABS(element) .eq. 0.0d0) goto 3
            if ((ABS(element) .eq. 0.0d0) .and. (i .ne. j)) goto 3
            if (next .gt. nzmax) then
               ierr = i
               return
            endif
            ja(next) = j
            a(next) = element
            next = next+1
 3       continue	   
         ia(i+1) = next
 4    continue
      return
c---- end of dnscsr ---------------------------------------------------- 
c----------------------------------------------------------------------- 
      end
c----------------------------------------------------------------------- 

