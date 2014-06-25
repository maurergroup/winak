        module ainvsr2module
            integer, allocatable, dimension(:) :: ip, jp
            double precision, allocatable, dimension(:) :: ap
        contains

* (c) sparslab module name=ainvsr2
*
* purpose:
*  symmetric right-looking ainv stabilized preconditioner construction.
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
*   original version for sparslab - tu - 07/03/1996.
*
* references:
*   benzi/meyer/tuma, 1996
*
c
c make_explicit_interface
c
      subroutine ainvsr2(msglvl,msgunit,n,ia,ja,a,
     *  size_p,size_c,size_r,diagtol,
     *  drfl,mi,diag_one,droptyp,imodif,fill,fillmax,
     *  ifillmax,garrow,garcol,info)


c
c parameters
c
      integer msglvl,msgunit,n,size_r,size_c,size_p
      integer ia(*),ja(*)
      double precision a(*)
      integer garcol,garrow,droptyp
      integer imodif,diag_one,fill,fillmax,ifillmax,info
      double precision mi,drfl,diagtol
c
c includes
c
c     include 'ainvini1.h'
c     include 'rank1n.h'
c     include 'cd2cs_f90.h'
c
c internals
c
      integer, pointer :: ptr(:),cnr(:),lenr(:),ptc(:),cnc(:),lenc(:)
      double precision, pointer :: h(:),diag(:)
      integer, allocatable :: wn01(:),wn02(:),wn03(:),wn04(:)
      double precision, allocatable :: wr01(:),wr02(:),wr03(:)
c
      integer ierr,gillmur,ind2,ind3,indexr,indexc
      integer ifr,iendru,iendlu,eps,idistr,idistc,lsize
      integer nit,savefill,nreallocr,nreallocc,nja
      double precision size_mult,dv1,one,zero,delta
      parameter(size_mult=1.5d0,one=1.0d0,zero=0.0d0)
      parameter(eps=2.2d-16,delta=1.0d-60)
c
c start of ainvsr2
c
c  -- allocation
c
c     write(*,*) 'life is good'
      nja=ia(n+1)-1
      size_r=max(3*nja,size_r)
      size_c=max(3*nja,size_c)
      allocate(wn01(n+1),wn02(n+1),wn03(n+1),wn04(n+1),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','ainvsr2',
     *  ' allocation error',8,2)
      allocate(wr01(n+1),wr02(n+1),wr03(n+1),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','ainvsr2',
     *  ' allocation error',8,2)
      allocate(ptr(n+1),lenr(n+1),cnr(size_r),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','ainvsr2',
     *  ' allocation error',8,2)
      allocate(ptc(n+1),lenc(n+1),cnc(size_c),h(size_c),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','ainvsr1',
     *  ' allocation error',8,2)
      size_r=size_r
c
c  -- allocate reallocatables
c
      allocate(diag(n),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','ainvsr2',
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
      diagtol=eps
      drfl=drfl+delta
      droptyp=droptyp
      call iwset(n,0,wn01,1)
      call iwset(n,0,wn04,1)
      call dwset(n,zero,wr01,1)
c
c  -- initialize the preconditioner using dynamic matrix formats
c  -- dsr and dsc.
c
      call ainvini1(n,n,ia,ja,ptc,cnc,lenc,h,size_c,
     *  ptr,cnr,lenr,size_r,
     *  iendru,iendlu,idistr,idistc,lsize,ierr)
      write(*,*) 'life is good'
c
c  -- set initial fill-in parameters
c
      fillmax=0
      ifr=n
      fill=n
c
c  -- echo
c
      if(msglvl.ge.2)
     *    write(*,*)
     *      ' compute symmetric right-looking ainv preconditioner'
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
     *    wn01,wr01,wn02,wr02,wn03,wn04,ind2,indexr,dv1,drfl)
c
c    -- pivot modification and computation of updating vectors
c
        call aicol(ptc(1),lenc(1),cnc(1),h(1),wn03,wr03,ind3,
     *    wr02,ind2,indexc,dv1,diagtol,imodif,gillmur,mi,diag_one)
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
          call serr(0,2,'sparslab','ainvsr2_s',
     *      ' matrix is structurally singular',1041,2)
        end if
c
c    -- test whether to update
c
        if(nit.lt.n) then
c
c      -- compute size of fill-in
c
          ifr=ifr-lenc(indexc)
          savefill=fill
c
c      -- echo
c
          if(msglvl.ne.0) then
            if(msglvl.eq.2) then
              if(nit/100.eq.dble(nit)/100.0d0) then
                if(msgunit.eq.0) then
                  write(*,*)' nit =',nit, ' fill = ',fill
                elseif(msgunit.gt.0) then
                  write(msgunit,*)' nit =',nit, ' fill = ',fill
                end if
              end if
            elseif(msglvl.ge.3) then
              if(msgunit.eq.0) then
                write(*,*)' nit =',nit, ' fill = ',fill
              elseif(msgunit.gt.0) then
                write(msgunit,*)' nit =',nit, ' fill = ',fill
              end if
            end if
          end if
c
c    -- sparse rank-one update
c
          call rank1n(msglvl,n,n,ptr,cnr,lenr,size_r,
     *      ptc,cnc,lenc,h,size_c,
     *      indexc,wr01,wn03,wr03,ind3,wn02,wr02,ind2,
     *      iendru,iendlu,drfl,nit,fill,idistc,
     *      garcol,garrow,droptyp,nreallocr,nreallocc)
c
          ifr=ifr+fill-savefill
c
c      -- update fillmax - maximum size of reduced fill-in
c      -- which is obtained in the iteration ifillmax during the
c      -- preconditioner computation
c
          if(fillmax.lt.ifr) then
            ifillmax=nit
            fillmax=ifr
          endif
          nit=nit+1
          go to 333
        end if
c
c  -- format transformation
c
      size_p=fill
      allocate(ip(n+1),jp(size_p),ap(size_p),stat=ierr)
      if(ierr.ne.0) call serr(0,2,'sparslab','ainvsr2',
     *  ' allocation error',8,2)
      call cd2cs_f90(n,ptc,diag,cnc,lenc,h,ip,jp,ap,diag_one)
c
c  -- deallocation
c
      deallocate(wn01,wn02,wn03,wn04)
      deallocate(wr01,wr02,wr03,diag)
      deallocate(ptr,cnr,lenr,ptc,cnc,lenc,h)
c
c  -- return
c
      return
c
c end of ainvsr2
c
      end subroutine

c     include 'sparslab.f'
        end module ainvsr2module
