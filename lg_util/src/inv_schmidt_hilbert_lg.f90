!dk, schmidt_hilbert

  subroutine inv_schmidt_hilbert_lg &
               (epsilon,MXD1,MXD2,ndim1,ndim2 &
                ,array,ainv,iarb,arb,work)

! #####################################################################
!
!     PURPOSE -
!
!        Invert general array using Schmidt-Hilbert Orthonormalization
!        (can be singular, non-symmetric, non-square array).
!
!        subroutine to apply this inverse to a vector
!        also included below (solv_Mxb_schmidt_hilbert_lg)
!
!     INPUT ARGUMENTS -
!
!        epsilon - Directions which occur in array with
!                      norm / (typical norm) < epsilon
!                  are considered part of the "arbitrary" space
!                  (hence epsilon is a relative accuracy;
!                   see below for "typical norm")
!        MXD1,MXD2,ndim1,ndim2
!                - MAX/actual dimenions of array,ainv
!        array   - the array to invert
!
!     WORKSPACE ARGUMENTS -
!
!        work    - workspace (contents ignored)
!
!     OUTPUT ARGUMENTS -
!
!        ainv - the inverse of array
!        iarb, arb -
!             iarb(k)=0 if the matrix is not singular
!                    >0 if the matrix is singular
!                       and an arbitrary amount of arb(k,i)
!                       can be added to soln(i)
!                       [ arb(k,i) is normalized ]
!                    <0 procedure yield iarb>0,
!                       but test |array*arb| < eps failed 
!                       [ if |arb| non-zero, it is normalized after test ]
!
!     PROGRAMMING NOTES -
!
!        logical sizes for eps are
!           (epsilon)*[ typical sqrt(sum_j array(j,i)^2) ]
!        or (epsilon)*[ typical entry of array ]
!        where epsilon is the desired (input) relative accuracy.
!        The trade off is how "localized eigenvectors" are
!        weighted (already-diagonal entries will need to have
!        a comparable size to the "typical" entry).
!        For now, the first form is used.
!
!        For matricies with both "large" and "small" entries,
!        where the eignevectors for each are desired, a different
!        routine, such as LUD decomposition, is required.
!
!     CHANGE HISTORY -
!
!     $Log: inv_schmidt_hilbert_lg.f,v $
!     Revision 2.00  2007/11/03 00:49:11  spchu
!     Import to CVS
!  
!PVCS    
!PVCS       Rev 1.2   Mon Oct 25 16:30:14 1999   jtg
!PVCS    fixed missing scale in epsilon test in solv routine
!PVCS    
!PVCS       Rev 1.1   Tue Oct 19 14:57:20 1999   jtg
!PVCS    fixed local_debug flag (was non-zero)
!PVCS    
!PVCS       Rev 1.0   Tue Oct 19 13:05:52 1999   jtg
!PVCS    Initial revision.
!
! ######################################################################
  implicit none

  integer ndim1,ndim2,MXD1,MXD2
  real*8 eps,epsilon
  real*8 array(MXD1,ndim2),ainv(MXD2,ndim1) &
               ,arb(MXD2,ndim2),work(MXD2,ndim1)
  integer iarb(ndim2)

  integer i,j,k,l,local_debug,iway
  real*8 xnorm,dotij,qq,qqq
! ........................................................................
  local_debug=0

  if (ndim2<=0.or.ndim2>MXD2) goto 9999
  if (ndim1<=0.or.ndim1>MXD1) goto 9999

! calculate eps with units to use in calculation
  eps=0.d0
  iway=1
  if (iway==1) then
     ! compare to "typical" (max) norm
     do i=1,ndim1
       qq=0.d0
       do j=1,ndim2
         qqq=array(i,j)
         qq=qq+qqq*qqq
       end do
       if (qq>eps) eps=qq
     end do
  else
     ! compare to "typical" (max) entry
     do i=1,ndim1
       do j=1,ndim2
         qqq=abs(array(i,j))
         if (qqq>eps) eps=qqq
       end do
     end do
  end if
  eps=eps*epsilon

! initialize ainv
  do i=1,ndim2
    do j=1,ndim1
      ainv(i,j)=0.d0
    end do
  end do

  do i=1,ndim2

! initialize work vector
    do j=1,ndim1
      work(i,j)=array(j,i)
    end do

! make work vector normal
    xnorm=0.d0
    do j=1,ndim1
      xnorm=xnorm+work(i,j)*work(i,j)
    end do
    xnorm=sqrt(xnorm)

    if (xnorm<=eps) then

! no vector: set iarb so don't use
      iarb(i)=1

    else

! initialize arb
      do j=1,ndim1
        work(i,j)=work(i,j)/xnorm
      end do
      do j=1,ndim2
        arb(i,j)=0.d0
      end do
      arb(i,i)=1.d0/xnorm

! make ith work vector normal to all below it
      do j=1,i-1
        if (iarb(j)==0) then
          dotij=0.d0
          do l=1,ndim1
            dotij=dotij+work(i,l)*work(j,l)
          end do
          do l=1,ndim1
            work(i,l)=work(i,l)-dotij*work(j,l)
          end do
          do l=1,ndim2
            arb(i,l)=arb(i,l)-dotij*arb(j,l)
          end do
        end if
      end do

! re-normalize ith work vector
      xnorm=0.d0
      do l=1,ndim1
        xnorm=xnorm+work(i,l)*work(i,l)
      end do
      xnorm=sqrt(xnorm)

      if (xnorm<=eps) then

! if no new piece of vector: set iarb so don't use
        iarb(i)=2

      else

! update work, arb
        iarb(i)=0
        do l=1,ndim1
          work(i,l)=work(i,l)/xnorm
        end do
        do l=1,ndim2
          arb(i,l)=arb(i,l)/xnorm
        end do

! increment ainv
        do k=1,ndim2
          do j=1,ndim1
            ainv(k,j)=ainv(k,j)+work(i,j)*arb(i,k)
          end do
        end do

      end if

    end if

  end do

! should have ainv at this point, plus allowed arbitrary vectors
! test and normalize allowed arbitrary vectors
! set flag to negative if fail

  do i=1,ndim2

    if (iarb(i)==0) then

      ! zero arb in case used accidentally
      do j=1,ndim2
        arb(i,j)=0.d0
      end do

    else if (iarb(i)==1) then

      ! no vector in this direction: set arb to delta(i,j)
      do j=1,ndim2
        arb(i,j)=0.d0
      end do
      arb(i,i)=1.d0

    else ! if (iarb(i)==2) then

      qq=0.d0
      do j=1,ndim2
        qq=qq+arb(i,j)*arb(i,j)
      end do
      qq=sqrt(qq)
      if (qq<=eps) then
        if (local_debug>0) &
               write(*,*) '*** iarb/=0, |arb|=0, not using i=',i
        iarb(i)=-1
        ! normalize if possible
        if (qq/=0.d0) then
           qq=1.d0/(qq)
           do j=1,ndim2
             arb(i,j)=arb(i,j)*qq
           end do
        end if
      else
        qq=1.d0/(qq)
        do j=1,ndim2
          arb(i,j)=arb(i,j)*qq
        end do
        qqq=0.d0
        do k=1,ndim1
          qq=0.d0
          do j=1,ndim2
            qq=qq+array(k,j)*arb(i,j)
          end do
          qqq=qqq+qq*qq
        end do
        qqq=sqrt(qqq)
        if (qqq>eps) then
          if (local_debug>0) write(*,*) &
                 '*** iarb/=0, |M(arb)|/=0, not using i=',i
          iarb(i)=-1
        end if
      end if
      if (local_debug>0) then
        write(*,*) 'zero eigenvalue vector ',i*iarb(i)
        write(*,10) (arb(i,j),j=1,ndim2)
      end if

    end if

  end do

! count indep vectors and print ainv
  if (local_debug>0) then
    j=0
    do i=1,ndim2
      if (iarb(i)==0) j=j+1
    end do
    write(*,*) 'a^T has',j,' independent vectors'
    write(*,*)'ainv'
    do i=1,ndim2
      write(*,10)(ainv(i,j),j=1,ndim1)
    end do
  end if

  ! sucessful return
  return

  ! error return
9999 continue
  if (local_debug>1) stop 'schmidt_hilbert_inv_lg error'
  call writloga('default',0, &
       'Warning: schmidt_hilbert_inv_lg: array dimensions not allowed' &
       ,0,i)
  return
!
! Format statement
!
10 format(1x,4g13.4)

  end

! ===========================================
  subroutine solv_Mxb_schmidt_hilbert_lg &
               (epsilon,MXD1,MXD2,ndim1,ndim2 &
                ,array,vec,sol,ainv,iarb,arb,work)

! #####################################################################
!
!     PURPOSE -
!
!        solve general Mx=b using Schmidt-Hilbert Orthonormalization
!        (M can be singular, non-symmetric, non-square array).
!
!     ARGUMENTS same as for inv_schmidt_hilbert_lg -
!
!        epsilon, MXD1,MXD2,ndim1,ndim2
!        array, work, ainv, iarb, arb
!
!     ADDITIONAL ARGUMENTS -
!
!        vec - the (input) b vector in MX=b
!        sol - the (output) x solution vector to MX=b
!
!     PROGRAMMING NOTES -
!
!        after calling inv_schmidt_hilbert_lg, the solution
!        is optimized (minimal magnitude, best fit)
!        wrt the arbitrary vectors. This is probably not necessary.
!
!     CHANGE HISTORY -
!
!        $Log: inv_schmidt_hilbert_lg.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!
! ######################################################################
  implicit none

  integer ndim1,ndim2,MXD1,MXD2
  real*8 epsilon
  real*8 array(MXD1,ndim2),ainv(MXD2,ndim1) &
               ,arb(MXD2,ndim2),work(MXD2,ndim1) &
               ,vec(ndim1),sol(ndim2)
  integer iarb(ndim2)

  integer i,j,local_debug
  real*8 qq,qqq,xerr,q1,q2,scale

! ........................................................................

  local_debug=0

  call inv_schmidt_hilbert_lg &
            (epsilon,MXD1,MXD2,ndim1,ndim2 &
            ,array,ainv,iarb,arb,work)

  do i=1,ndim2
    sol(i)=0.d0
    do j=1,ndim1
      sol(i)=sol(i)+ainv(i,j)*vec(j)
    end do
  end do

  if (local_debug>0) then
     write(*,*) (sol(i),i=1,ndim2)
  end if

  ! minimize magnitude wrt (normalized) arbitrary components
  ! since arb normalized, don't need epsilon test?
  do i=1,ndim2
    if (iarb(i)>0) then
      qqq=0.d0
      do j=1,ndim2
        qqq=qqq+arb(i,j)*sol(j)
      end do
      do j=1,ndim2
        sol(j)=sol(j)-arb(i,j)*qqq
      end do
    end if
  end do

  ! test that sol - if not, use best fit
  ! find scale for vec
  scale=0.d0
  do i=1,ndim1
     scale=scale+vec(i)*vec(i)
  end do

  ! test that sol - if not, use best fit
  xerr=0.d0
  q1=0.d0
  q2=0.d0
  do i=1,ndim1
    work(1,i)=0.d0
    do j=1,ndim2
      work(1,i)=work(1,i)+array(i,j)*sol(j)
    end do
    qq=work(1,i)-vec(i)
    xerr=xerr+qq*qq
    q1=q1+work(1,i)*vec(i)
    q2=q2+work(1,i)*work(1,i)
  end do
  if (xerr>epsilon*scale) then
     if (local_debug>0) &
            write(*,*) '*** NO sol WITH THIS vec ***'
     ! scale to best fit magnitude
     if (q2>epsilon*scale) then
       q1=q1/q2
       do i=1,ndim2
         sol(i)=sol(i)*q1
       end do
     else
       do i=1,ndim2
         sol(i)=0.d0
       end do
     end if
     xerr=0.d0
     do i=1,ndim1
       qq=0.d0
       do j=1,ndim2
         qq=qq+array(i,j)*sol(j)
       end do
       qq=qq-vec(i)
       xerr=xerr+qq*qq
     end do
  end if

  if (local_debug>0) then
     write(*,*)'(sol(i),i=1,',ndim2,')'
     write(*,10)(sol(i),i=1,ndim2)
     write(*,*)'xerr=',xerr
  end if

9999 continue

  return
!
! Format statement
!
10 format(1x,4g13.4)

   end subroutine solv_Mxb_schmidt_hilbert_lg

! ===========================================
