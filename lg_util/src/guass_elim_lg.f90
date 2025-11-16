!dk, guass_elim_lg
  subroutine guass_elim_lg(flag,n,LDA,epsilon,A,b,x &
                                 ,wrkmat,ierror)
 
! #####################################################################
!
!     PURPOSE -
!
!        solve general Ax=b using Gaussian elimination.
!        (A can be singular and non-symmetric, but must be square).
!
!     INPUT ARGUMENTS -
!
!        A - the array to invert
!        LDA - MAX first dimension of A to get storage correct
!        n - dimension of problem: A(1:n,1:n)*x(1:n)=b(1:n)
!        b - the (input) b vector in Ax=b
!        epsilon - relative accuracy
!             The average eigenvalue of b is computed as |Ab|/|b|,
!             and hence |x|~|b|/(|Ab|/|b|). Corrections dx to the estimate along
!             eigendirections of A with significantly smaller eigenvalues
!             give large changes to x for negligible improvement to |Ax-b|.
!             epsilon is used for the relative tests as appropriate.
!        flag - switch to indicate whether to calculate solution of
!               (1) Ax=b
!               (2) (A^2+eps^2*I)x=Ab where I(i,j)=(1 if i=j, 0 otherwise)
!                   [solve (A+i*eps)x=b in sense of Hadamard principle part]
!
!     OUTPUT ARGUMENTS -
!
!        x - the (output) solution vector x to Ax=b
!        ierror - set on return to
!                -1         if A=0 and b.ne.0
!                 0         if (|Ax-b|/|b|)^2 < epsilon,
!                 1+min((|Ax-b|/|b|)^2/(10*epsilon),1000000)
!                           otherwise
!                 |Ax-b|/|b| < epsilon should also hold
!                 but using this softer test for the error return.
!
!     WORKSPACE ARGUMENTS -
!
!        wrkmat - nxn matrix workspace (contents ignored)
!
!     SPELLING NOTES -
!
!        oops: Gauss is spelled "gauss" not "guass",
!        but since I checked in in that way I guess it'll stay ...
!
!     PROGRAMMING NOTES -
!
!        The basic code is a straightforward implementation of
!        Gaussian elimination. The modifications here are
!        giving the user the choice to solve (A^2+eps*I)x=Ab
!        (Hadamard principle part) for (nearly) singular matricies,
!        and putting in the epsilon tests (and calculating the scale
!        for epsilon). As it is a principle part, flag=2 approaches
!        epsilon=0 as a well defined limit, whereas flag=1 can be
!        "unpredictable" especially for the nearly singular case.
!
!        For nearly singular matricies where "wild" contributions (essentially
!        unrestricted small eigenvalue components) to x are undesireable,
!        using flag=2 and epsilon "relatively large" (1.e-2 to 1.e-4)
!        is recommended. This also works pretty well for the non-singular cases.
!
!        Using flag=1 (straightforward Ax=b solution)
!        has accuracy problems in certain cases, eg for the 2x2 problem
!           ( 0 1 ) (x1)   ( 1 )           (x1)   (0.999977878280)
!           ( 1 0 ) (x2) = ( 1 )  one gets (x2) = (0.999999999999) for epsilon=1.e-12;
!        flag=2 is more stable, although for large matricies it is slightly
!        more inefficient since the matrix has to be squared. (There is one
!        other "N^3" loop in this code, so it is not a huge overhead compare to
!        flag=1 -- the coding presumption is that the dimension of the matrix A
!        is typically n=3 and not n=1000...)
!        However, note that for flag=1 "often" |Ax-b|/|b| << epsilon,
!        whereas for flag=2 |Ax-b|/|b| ~ epsilon "almost always".
!
!        Although not coded here, a "best" epsilon could also be selected
!        by minimizing a functional showing a minimum and having the
!        appropriate symmetry properties, such as
!            chisq=|x|^2*(Tr A^2 + 3 eps^2)^2
!        or looking to see if/where |x|-|b|/(|Ab|/|b|) change sign.
!        (These are 2 empirical tests that worked fairly well for
!        the set of small matricies I looked at, which came from
!        typical cases projecting face velocity onto node velocity.)
!
!        Also compare solv_Mxb_schmidt_hilbert_lg (forms inverse with Schmidt-Hilbert).
!        For large matrices, using an iterative sparse solver such as GMRES is recommended
!
!     CHANGE HISTORY -
!
!        $Log: guass_elim_lg.f,v $
!        Revision 2.00  2007/11/03 00:49:10  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.7   17 Feb 2001 14:19:24   jtg
!PVCS    fixed log line; Rev 1.6=correctly did Rev 1.5
!PVCS    
!PVCS       Rev 1.5   Wed Mar 08 15:40:30 2000   dcg
!PVCS    change debug flag so that error messages are not written
!PVCS    return flag will have error status
!PVCS
!PVCS       Rev 1.2   Mon Dec 13 10:47:16 1999   jtg
!PVCS    Noted Gauss is spelled wrong
!PVCS
!PVCS       Rev 1.1   Wed Dec 01 15:43:52 1999   jtg
!PVCS    set local_debug to 0
!PVCS
!PVCS       Rev 1.0   Wed Dec 01 14:35:32 1999   jtg
!PVCS    Initial revision.
!
! ######################################################################
 
  implicit none
 
  integer n,LDA,flag,ierror
 
  real*8 x(n),b(n),A(LDA,n),wrkmat(n,n) &
             ,epsilon
 
  integer i,j,k,local_debug
  real*8 eps,b2,Ab2,xxx
  character*132 cbuf
 
! ---------------------------------------------------------------
  local_debug=0
 
! ...............................
! fill wrkmat,x from A,b
 
  if (flag==2) then
     ! set wrkmat=A^2, x=Ab
     ! obviously wrkmat and A, and x and b, need to be separate storage
     ! locations for this flag case.
     do i=1,n
        x(i)=0.d0
        do j=1,n
           x(i)=x(i)+A(i,j)*b(j)
           wrkmat(i,j)=0.d0
           do k=1,n
              wrkmat(i,j)=wrkmat(i,j)+A(i,k)*A(k,j)
           end do
        end do
     end do
  else
     ! set wrkmat=A, x=b
     ! except for the final test comparing |Ax-b|/|b|,
     ! wrkmat and A, and x and b, could have been indentical storage locations.
     do i=1,n
        x(i)=b(i)
        do j=1,n
           wrkmat(i,j)=A(i,j)
        end do
     end do
  end if
 
! ...............................
! figure out eps from epsilon and |Ab|/|b|
! eps is used to screen out singular components
 
! calculate scale of A and b
  Ab2=0.d0
  b2=0.d0
  do i=1,n
     b2=b2+x(i)*x(i)
     xxx=0.d0
     do j=1,n
        xxx=xxx+wrkmat(i,j)*x(j)
     end do
     Ab2=Ab2+xxx*xxx
  end do
 
! abort if solution trivial (b=0), or no soln possible (A=0)
  if (b2==0.d0 .or. Ab2==0.d0) then
     do i=1,n
        x(i)=0.d0
     end do
     goto 100
  end if
 
! set absolute eps for iteration, update diagonal for flag=2
  xxx=sqrt(Ab2/b2)
  eps=epsilon*xxx
  if (flag==2) then
     ! vs using xxx=Tr(wrkmat)/dble(n) (average eigenvalue of A^2)
     ! if along small eigenvalue direction, I am presuming current
     ! xxx=sqrt(Ab2/b2) (average eigenvalue component in Ab) better ...
     xxx=eps*epsilon   ! wrkmat = ( A^2 + epsilon^2 * scale(A^2) )
     do i=1,n
        wrkmat(i,i)=wrkmat(i,i)+xxx
     end do
     ! solve as for unsquared case, but with smaller epsilon?
     ! which version to use ... does it make a difference?
     ! eps=eps
     eps=1.d-4*eps
     ! eps=epsilon*eps
  end if
 
! ...............................
! solve using Gaussian elimination,
! testing the diagonals of A against eps to avoid dividing by zero
 
  do k=1,n-1
     do i=k+1,n
        xxx=wrkmat(k,k)
        if (abs(xxx)>eps) then
           xxx=1.d0/xxx
        else if (xxx<0.d0.and.flag/=2) then
           xxx=-1.d0/eps
        else
           xxx=1.d0/eps
        end if
        xxx=xxx*wrkmat(i,k)
        x(i)=x(i)-x(k)*xxx
        do j=k+1,n
           wrkmat(i,j)=wrkmat(i,j)-wrkmat(k,j)*xxx
        end do
     end do
  end do
 
  do i=n,1,-1
     do j=i+1,n
        x(i)=x(i)-wrkmat(i,j)*x(j)
     end do
     xxx=wrkmat(i,i)
     if (abs(xxx)>eps) then
        xxx=1.d0/xxx
     else if (xxx<0.d0.and.flag/=2) then
        xxx=-1.d0/eps
     else
        xxx=1.d0/eps
     end if
     x(i)=x(i)*xxx
  end do
 
! ...............................
! test how good a solution and update ierror
! have to redo b2 since for flag=2 it is |Ab| rather than |b|
 
  ierror=0
 
  b2=0.d0
  Ab2=0.d0
  do i=1,n
     xxx=-b(i)
     do j=1,n
        xxx=xxx+A(i,j)*x(j)
     end do
     Ab2=Ab2+xxx*xxx
     b2=b2+b(i)*b(i)
  end do
 
  if (local_debug/=0) then
     write(*,'(a,4g20.10)') 'gauss_elim_lg: |Ax-b|,|b|,epsilon=' &
                 ,sqrt(Ab2),sqrt(b2),(sqrt(Ab2)/sqrt(b2))/epsilon &
                 ,epsilon
  end if
 
  if (b2/=0.d0) then
     ! actually, should test sqrt(Ab2/b2) against epsilon,
     ! but since this is a "hard error flag" (0 vs non-zero), use epsilon instead
     xxx=Ab2/b2
     if (epsilon/=0.d0) xxx=xxx/epsilon
     if (xxx>1.d0) then
        if (xxx>10000000.d0) then
           ierror=1000000
        else
           ierror=1+int(xxx*0.1d0)
        end if
     end if
  else if (Ab2/=0.d0) then
     ierror=-1
  end if
 
  if (ierror/=0.and.local_debug/=0) then
     write(cbuf,'(a,3g15.5)') &
               'gauss_elim_lg warning: (||Ax-b|/|b|)^2 > epsilon;' &
               //' |Ax-b|,|b|,epsilon=',sqrt(Ab2),sqrt(b2),epsilon
     call writloga('default',0,cbuf,0,i)
  end if
 
! ...............................
100 continue
  return
   end subroutine guass_elim_lg
 
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
