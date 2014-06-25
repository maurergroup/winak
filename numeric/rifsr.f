        module rifsrmodule
            integer, allocatable, dimension(:) :: ip, jp
            double precision, allocatable, dimension(:) :: ap
        contains

* (c) sparslab module name=rifsr
*
* purpose:
*  symmetric right-looking rif preconditioner based on
*  symmetric sainv preconditioner construction.
*  factorized approximate inverse is in the form: a^{-1}=z^t*d^{-1}*z,
*  where z is the lower triangular matrix stored by rows.
*
*       | *  *  .  .  .  |   | *              |   | *              |
*       |    *  .  .  ^  |   |    *           |   | *  *           |
*       |       *  .  |  | * |       *        | * | .  .  *        |
*       |          *  |  |   |          *     |   | .  .  .  *     |
*       |             *  |   |             *  |   | .  ---->    *  |
*
* a^{-1} =     z^t        *         d^{-1}     *         z
*
* history:
*   original version for sparslab - tu - 20/08/2000.
*
* parameters:
*
*   msglvl  (integer; input)  level of output messages
*                    =0  for no messages
*   msgunit  (integer; input)  unit number for output messages
*   n,ia(n+1),ja(ia(n+1)-1),a(ia(n+1)-1)
*       (integer,integer,integer,double; input)  spd
*        matrix to be factorized in csr matrix format 11
*   size_c  (integer; input)  starting size for storing column
*        information in dynamic matrix storage scheme
*   size_r  (integer; input)  starting size for storing row
*        information in dynamic matrix storage scheme
*   ip(n+1),jp(ip(n+1)-1),ap(ip(n+1)-1)
*        (integer pointer,integer pointer,double pointer; output)
*        spd rif decomposition in format 311
*   size_p  (integer; input)  starting size for storing output rif
*        factor
*   diagtol (double; input) tolerance for size of diagonal entries;
*        smaller entries are replaced by diagtol
*   drflic (double; input) drop tolerance for the rif factors
*   drflai (double; input) drop tolerance for the sainv generating process
*   mi (double; input) threshold for computation of the rif factors
*        internal value
*        =0.1d0
*   diag_one (integer; input) a parameter for output control
*        internal value
*        =1
*   droptyp (integer; input) dropping parameter controlling
*        absolute or relative dropping
*        internal value
*        =0
*   imodif (integer; output) number of modifications of diagonal
*        entries
*   filldyn (integer; output) maximum fill obtained sainv process
*   fillmax (integer; output) maximum total (sum of fill in rif
*        and in sainv) fill obtained during the process
*   ifillmax (integer; output) index where the maximum total fill
*        was obtained
*   garrow (integer; output) number of row garbage collections
*   garcol (integer; output) number of column garbage collections
*   fillic (integer; output) size of the rif factor
*   info (integer; output) output info; should be 0
*
c
c make_explicit_interface
c
      subroutine rifsr(msglvl,msgunit,n,ia,ja,a,size_c,
     *  size_r,size_p,
     *  diagtol,drflic,drflai,mi,diag_one,droptyp,imodif,
     *  filldyn,fillmax,
     *  ifillmax,garrow,garcol,fillic,info)
c
c parameters
c
      integer n,msglvl,msgunit,size_r,size_c,size_p
      integer garcol,garrow,droptyp,fillic
      integer imodif,diag_one,filldyn,fillmax,ifillmax
      integer info
      integer ia(*),ja(*)
      double precision a(*)
      double precision mi,drflai,drflic,diagtol
c
c includes
c
c     include 'ainvini1.h'
c     include 'rank1n2.h'
c
c internals
c
      integer, pointer :: ptr(:),cnr(:),lenr(:)
      integer, pointer :: ptc(:),cnc(:),lenc(:),im(:),jm(:)
      double precision, pointer :: h(:),am(:),diag(:)
      integer, allocatable :: wn01(:),wn02(:),wn03(:),wn04(:)
      double precision, allocatable :: wr01(:),wr02(:),wr03(:)
      integer, allocatable :: jp_new(:)
      double precision, allocatable :: ap_new(:)
c
      integer ierr
      integer iendru,iendlu,i
      integer totalfill,nit,indp,fill
      integer ind2,indexr,indexc,ind3
      double precision dv1
      integer idistr,idistc,lsize,gillmur,nreallocr,nreallocc
      double precision eps,zero,one,delta,temp
      parameter(gillmur=0)
      parameter(eps=2.2d-16,zero=0.0d0,one=1.0d0,delta=1.0d-60)
c
c start of rifsr
c
c  -- allocation
c
c
      allocate(wn01(n+1),wn02(n+1),wn03(n+1),wn04(n+1),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *  ' allocation error',8,2)
      allocate(wr01(n+1),wr02(n+1),wr03(n+1),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *  ' allocation error',8,2)
      allocate(ptr(n+1),lenr(n+1),cnr(size_r),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *  ' allocation error',8,2)
      allocate(ptc(n+1),lenc(n+1),cnc(size_c),h(size_c),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *  ' allocation error',8,2)
c
c  -- allocate reallocatables
c
      allocate(ip(n+1),jp(size_c),ap(size_c),diag(n),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *  ' allocation error',8,2)
      allocate(im(n+1),jm(size_c),am(size_c),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *  ' allocation error',8,2)
c
c  -- initialization
c
      info=0
      idistr=1
      idistc=1
      lsize=5
      garcol=0
      garrow=0
      imodif=0
      nit=1
      indp=1
      ip(1)=1
      diagtol=eps
      drflai=drflai+delta
      droptyp=droptyp
      call iwset(n,0,wn01,1)
      call iwset(n,0,wn04,1)
      call dwset(n,zero,wr01,1)
c
c  -- initialize the preconditioner using dynamic matrix formats
c  -- dsr and dsc.
c
      print*, ia(n), ja(ia(n)-1), a(1), a(ia(n)-1)
      call ainvini1(n,n,ia,ja,ptc,cnc,lenc,h,size_c,
     *  ptr,cnr,lenr,size_r,
     *  iendru,iendlu,idistr,idistc,lsize,ierr)
c
c  -- set initial fill-in parameters
c
      fillmax=0
      fill=n
c
c  -- echo
c
      if(msglvl.ge.1)
     *    write(*,*)'   compute symmetric right-looking rif preconditi',
     *'oner'
c
c  -- the loop nit=1 ...
c
  333 continue
c
c    -- find a pivot
c
        indexr=nit
        indexc=nit
        call ainvrow2(n,ia,ja,a,ptc(1),lenc(1),cnc(1),h(1),
     *    ptr(1),lenr(1),cnr(1),
     *    wn01,wr01,wn02,wr02,wn03,wn04,ind2,indexr,dv1,drflai)
c
c    -- pivot modification and computation of updating vectors
c
        call aicol(ptc(1),lenc(1),cnc(1),h(1),wn03,wr03,ind3,
     *    wr02,ind2,indexc,dv1,diagtol,imodif,gillmur,mi,diag_one)

        if(indp-1+ind2.ge.size_p) then
          size_p=size_p+min(size_p,(n*n+1-nit*(nit-1))/2)
          allocate(ap_new(size_p),stat=ierr)
          if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *      ' missing space to store the preconditioner',1603,2)
          ap_new(1:size_p)=ap(1:size_p)
          deallocate(ap)
          allocate(ap(size_p),stat=ierr)
          if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *      ' missing space to store the preconditioner',1603,2)
          ap(1:size_p)=ap_new(1:size_p)
          deallocate(ap_new)
          allocate(jp_new(size_p),stat=ierr)
          if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *      ' missing space to store the preconditioner',1603,2)
          jp_new(1:size_p)=jp(1:size_p)
          deallocate(jp)
          allocate(jp(size_p),stat=ierr)
          if(ierr.ne.0) call serr(0,2,'sparslab','rifsr',
     *      ' missing space to store the preconditioner',1603,2)
          jp(1:size_p)=jp_new(1:size_p)
          deallocate(jp_new)
        end if
        jp(indp)=nit
        ap(indp)=dv1
        indp=indp+1
        do i=1,ind2
          temp=abs(wr02(i))
          if(temp.ge.drflic) then
            jp(indp)=wn02(i)
            ap(indp)=wr02(i)
            indp=indp+1
          end if
        end do
        ip(nit+1)=indp
c
c    -- save diagonal factor
c    -- this block can be considered as an anachronism because in case
c    -- that diag_one.eq.1 this array is not used and in the opposite
c    -- case use of diag can be avoided by setting diagonal elements
c    -- in factor. but we keep that as an monument of the old style
c    -- routines until new classification of preconditioner formats
c    -- is in all the routines.
c
        if(diag_one.eq.1) then
          diag(nit)=one
        else
          diag(nit)=dv1
        end if
c
c    -- test numerical and structural singularity
c
        if(indexr.le.0.or.indexc.le.0) then
          call serr(0,2,'sparslab','rifsr',
     *      ' matrix is structurally singular',1041,2)
        end if
c
c    -- test whether to update
c
        if(nit.lt.n) then
c
c      -- echo
c
          if(msglvl.ne.0) then
            if(msglvl.eq.2) then
              if(nit/100.eq.dble(nit)/100.0d0) then
                if(msgunit.eq.0) then
                  write(*,*)' nit =',nit, ' fill_sainv = ',fill,
     *              ' fill_ic = ',indp-1
                elseif(msgunit.gt.0) then
                  write(msgunit,*)' nit =',nit, ' fill_sainv = ',fill,
     *              ' fill_ic = ',indp-1
                end if
              end if
            elseif(msglvl.ge.3) then
              if(msgunit.eq.0) then
                write(*,*)' nit =',nit, ' fill_sainv = ',fill,
     *            ' fill_ic = ',indp-1
              elseif(msgunit.gt.0) then
                write(msgunit,*)' nit =',nit, ' fill_sainv = ',fill,
     *            ' fill_ic = ',indp-1
                end if
            end if
          end if
c
c    -- sparse rank-one update
c
          call rank1n2(msglvl,n,n,ptr,cnr,lenr,
     *      size_r,ptc,cnc,lenc,h,size_c,
     *      indexc,wr01,wn03,wr03,ind3,wn02,wr02,ind2,
     *      iendru,iendlu,drflai,nit,fill,idistc,
     *      garcol,garrow,droptyp,nreallocr,nreallocc)
c
          totalfill=fill+indp-1
c
c      -- update fillmax - maximum size of reduced fill-in
c      -- which is obtained in the iteration ifillmax during the
c      -- preconditioner computation
c
          if(fillmax.lt.totalfill) then
            ifillmax=nit
            fillmax=totalfill
            filldyn=fill
          endif
          nit=nit+1
          go to 333
        end if
        fillic=indp-1
c
c  -- deallocation
c
      deallocate(wn01,wn02,wn03,wn04)
      deallocate(wr01,wr02,wr03)
      deallocate(ptr,cnr,lenr,diag)
c
c  -- return
c
      return
c
c end of rifsr
c
      end subroutine

c     include 'sparslab.f'
        end module rifsrmodule
