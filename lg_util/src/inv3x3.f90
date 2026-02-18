!dk,inv3x3
subroutine inv3x3(a11,a12,a13,a21,a22,a23,a31,a32,a33,b1,b2,b3, &
                       x1,x2,x3,iflag)
!
!
! #####################################################################
!
!     PURPOSE -
!
!        Solv the 3x3 problem A x = b for x
!
!     INPUT ARGUMENTS -
!
!        a11,a12,a13
!        a21,a22,a23
!        a31,a32,a33 - the A matrix
!        b1,b2,b3    - the b vector
!
!     OUTPUT ARGUMENTS -
!
!        x1,x2,x3 - the solution vector
!        iflag - 0 if matrix was not singular
!                1 if it was and so used "best" schmidt hilbert solution
!                  (optimized for smallest magnitude wrt arbitrary pieces,
!                   and best fit of Ax to b in case no valid solution)
!
!     PROGRAMMING NOTES -
!
!        "epsilon"'s hardwired: could pass...
!
!     CHANGE HISTORY -
!
!        $Log: inv3x3.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.7   Wed Dec 01 16:15:56 1999   jtg
!PVCS    changed to call Guassian elimination instead of Schmidt-Hilbert
!PVCS    
!PVCS       Rev 1.6   Mon Oct 25 17:11:54 1999   jtg
!PVCS    changed to using the "best" Schmidt-Hilbert solution
!PVCS    (optimized wrt arbitrary vectors, as opposed to 
!PVCS    simply using x=M^{-1}b) as this gave significantly more
!PVCS    "useful" results in test calculations.
!PVCS    
!PVCS       Rev 1.5   Tue Oct 19 15:00:12 1999   jtg
!PVCS    fixed local_debug flag (was non-zero)
!PVCS    
!PVCS       Rev 1.4   Tue Oct 19 13:01:42 1999   jtg
!PVCS    If the matrix is singular, it now calls Schidt-Hilbert routines
!PVCS    to find the best-fit magnitude. Also, if the determinant is
!PVCS    small compared to the contributions, it will compare the
!PVCS    magnitudes of the Schidt-Hilbert and Cramer solutions,
!PVCS    using the Cramer solution unless the solutions differ by
!PVCS    a few orders of magnitude, in which case it presumes that
!PVCS    what the user "really" wanted was the low-order Schidt-Hilbert
!PVCS    solution since "trying to get a really good answer" is probably
!PVCS    what caused the Cramer solution to have a large magnitude
!PVCS    in the "nearly singular" case.
!PVCS    
!PVCS       Rev 1.3   Fri Jan 29 09:28:46 1999   dcg
!PVCS    define xfuzz
!PVCS
!PVCS       Rev 1.2   11/16/95 17:10:14   het
!PVCS    Correct an error matrix inverter.
!PVCS
!PVCS       Rev 1.1   01/04/95 21:55:32   llt
!PVCS    unicos changes (made by het)
!PVCS
!PVCS       Rev 1.0   11/15/94 16:50:26   llt
!PVCS    Original version
!
! ######################################################################
!
implicit none
real*8 a11,a12,a13,a21,a22,a23,a31,a32,a33,b1,b2,b3, &
                       x1,x2,x3

integer iflag,i1,i2,j1,j2,i,j,ierr
!$$  &       ,local_debug,iarb(3)

real*8 mat(3,3),s(3),b(3),xxx,sss &
            ,work(3,3),scale,det,epsilon
!$$  &       ,matinv(3,3),arb(3,3)

!-----------------------------------------------------------------------

iflag=0

xxx=abs(b1)
if (xxx<abs(b2)) xxx=abs(b2)
if (xxx<abs(b3)) xxx=abs(b3)

! do nothing if vec==0
if (xxx==0.d0) then
   x1=0.d0
   x2=0.d0
   x3=0.d0
   goto 100
end if

mat(1,1)=a11
mat(2,1)=a21
mat(3,1)=a31
mat(1,2)=a12
mat(2,2)=a22
mat(3,2)=a32
mat(1,3)=a13
mat(2,3)=a23
mat(3,3)=a33

! calculate determinant and adjoint
scale=0.d0
det=0.d0
do i=1,3
   i1=1+mod(i,3)
   i2=1+mod(i1,3)
   xxx=mat(1,i)*mat(2,i1)*mat(3,i2)
   if (abs(xxx)>scale) scale=abs(xxx)
   det=det+xxx
   xxx=mat(1,i)*mat(2,i2)*mat(3,i1)
   if (abs(xxx)>scale) scale=abs(xxx)
   det=det-xxx
   do j=1,3
      i1=1+mod(i,3)
      i2=1+mod(i1,3)
      j1=1+mod(j,3)
      j2=1+mod(j1,3)
      work(j,i)=mat(i1,j1)*mat(i2,j2)-mat(i1,j2)*mat(i2,j1)
   end do
end do

! calculate soln using inverse=adjoint/determinant (Cramer's rule)

if (det/=0.d0) then
   x1=work(1,1)*b1+work(1,2)*b2+work(1,3)*b3
   x1=x1/det
   x2=work(2,1)*b1+work(2,2)*b2+work(2,3)*b3
   x2=x2/det
   x3=work(3,1)*b1+work(3,2)*b2+work(3,3)*b3
   x3=x3/det
end if

if (abs(det)<=1.d-2*scale.or.det==0.d0) then

   ! if (nearly) singular, compare to Schmidt-Hilbert soln and
   ! use that instead if magnitudes vastly different

   epsilon=1.d-4

!$$      ! calc Schmidt-Hilbert inv
!$$      call inv_schmidt_hilbert_lg
!$$  &          (epsilon,3,3,3,3
!$$  &           ,mat,matinv,iarb,arb,work)
!$$      ! apply to get soln
!$$      do i=1,3
!$$        s(i)=matinv(i,1)*b1+matinv(i,2)*b2+matinv(i,3)*b3
!$$      enddo

   b(1)=b1
   b(2)=b2
   b(3)=b3
!$       ! calc "best" Schmidt-Hilbert soln
!$       call solv_Mxb_schmidt_hilbert_lg
!$   &          (epsilon,3,3,3,3
!$   &           ,mat,b,s,matinv,iarb,arb,work)
   ! solve using Guassian elimination
   call guass_elim_lg(2,3,3,epsilon,mat,b,s,work,ierr)

   if (det/=0.d0) then
      xxx=abs(x1)+abs(x2)+abs(x3)
      sss=abs(s(1))+abs(s(2))+abs(s(3))
      if (xxx>1.d1*sss) then
         ! presume that what the user really wanted was
         ! the "principle part" (M^2+eps^2I)s=Mb soln of the
         ! (nearly) singular matrix
         x1=s(1)
         x2=s(2)
         x3=s(3)
         iflag=1 ! flag as nearly singular
         if (ierr/=0) iflag=1+abs(ierr)
      end if
   else
      x1=s(1)
      x2=s(2)
      x3=s(3)
      iflag=1 ! flag as singular
      if (ierr/=0) iflag=1+abs(ierr)
   end if

end if

!$$   local_debug=0
!$$   if (local_debug.gt.0) then
!$$     write(*,*) x1,x2,x3,a11*x1+a12*x2+a13*x3-b1
!$$  &            ,a21*x1+a22*x2+a23*x3-b2
!$$  &            ,a31*x1+a32*x2+a33*x3-b3
!$$   endif

100 return
   end subroutine inv3x3
