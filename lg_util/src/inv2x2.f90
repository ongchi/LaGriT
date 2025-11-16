!dk,inv2x2
subroutine inv2x2(a11,a21,a12,a22,x1,x2,b1,b2,epsilon)
!                                                                       
! ######################################################################
!                      
!     PURPOSE -
!
!        Solv the 2x2 problem a x = b for x
!
!     INPUT ARGUMENTS -
!
!        a11,a21,a12,a22 - the a matrix
!        b1,b2           - the b vector
!        epsilon         - relative epsilon for small tests
!
!     OUTPUT ARGUMENTS -
!
!        x1,x2 - the solution vector
!
!     PROGRAMMING NOTES -
!
!        to handle "nearly singular" matrix, the
!        Cramers Rule soln is compared to
!        the Schmidt-Hilbert soln.
!        If the magnitudes differ widly
!        the Schmidt-Hilbert soln is used.
!
!     CHANGE HISTORY -
!
!        $Log: inv2x2.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.3   Tue Oct 19 14:58:56 1999   jtg
!PVCS    fixed local_debug flag (was non-zero)
!PVCS    
!PVCS       Rev 1.2   Tue Oct 19 12:59:42 1999   jtg
!PVCS    If singular, it now finds the best solution instead of
!PVCS    returning "1/epsilon", using Schmidt-Hilbert to
!PVCS    solve the singular problem.
!PVCS
!PVCS       Rev 1.1   01/04/95 21:55:30   llt
!PVCS    unicos changes (made by het)
!PVCS
!PVCS       Rev 1.0   11/15/94 16:50:24   llt
!PVCS    Original version
!
! ######################################################################
!
implicit none
real*8 a11,a12,a21,a22,b1,b2, &
                       x1,x2,epsilon

integer iflag,i1,i,local_debug

real*8 mat(2,2),matinv(2,2),s1,s2,xxx,sss &
            ,iarb(2),arb(2,2),work(2,2),scale,det

!-----------------------------------------------------------------------

xxx=abs(b1)
if (xxx<abs(b2)) xxx=abs(b2)

! do nothing if vec==0
if (xxx==0.d0) then
   x1=0.d0
   x2=0.d0
   goto 100
end if

mat(1,1)=a11
mat(2,1)=a21
mat(1,2)=a12
mat(2,2)=a22

! calculate determinant and adjoint
scale=0.d0
det=0.d0
i=1
i1=2
xxx=mat(1,i)*mat(2,i1)
if (abs(xxx)>scale) scale=abs(xxx)
det=det+xxx
xxx=mat(1,i1)*mat(2,i)
if (abs(xxx)>scale) scale=abs(xxx)
det=det-xxx
matinv(1,1)=mat(2,2)
matinv(2,2)=mat(1,1)
matinv(1,2)=-mat(1,2)
matinv(2,1)=-mat(2,1)

! calculate soln using inverse=adjoint/determinant (Cramer's rule)

if (det/=0.d0) then
   x1=matinv(1,1)*b1+matinv(1,2)*b2
   x1=x1/det
   x2=matinv(2,1)*b1+matinv(2,2)*b2
   x2=x2/det
end if

if (abs(det)<=epsilon*scale.or.det==0.d0) then

   ! if (nearly) singular, compare to Schmidt-Hilbert soln and
   ! use that instead if magnitudes vastly different

   call inv_schmidt_hilbert_lg &
               (epsilon,2,2,2,2 &
                ,mat,matinv,iarb,arb,work)

   if (det/=0.d0) then
      s1=matinv(1,1)*b1+matinv(1,2)*b2
      s2=matinv(2,1)*b1+matinv(2,2)*b2
      xxx=abs(x1)+abs(x2)
      sss=abs(s1)+abs(s2)
      if (xxx*epsilon>sss) then
         ! presume that what the user really wanted was
         ! the schmidt_hilbert soln of the (nearly) singular matrix
         x1=s1
         x2=s2
         iflag=1
      end if
   else
      x1=matinv(1,1)*b1+matinv(1,2)*b2
      x2=matinv(2,1)*b1+matinv(2,2)*b2
   end if

end if

!local_debug=0
!if (local_debug.gt.0) then
!   write(*,*) x1,x2,a11*x1+a12*x2-b1,a21*x1+a22*x2-b2
!endif

100 return
   end subroutine inv2x2
