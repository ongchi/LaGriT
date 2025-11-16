!dk,kmprsz
subroutine kmprsz(n,z,iz,x,ix,y,iy,count)
! 
!                                                                      
! ##################################################################### 
!                                                                       
!     PURPOSE -                                                         
!                                                                       
!        Compress a vector x into vector y where z is equal to zero.                                                           
!                                                                       
!     INPUT ARGUMENTS -                                                 
!                                                                       
!        n   - (integer) the length of the input vector.
!        z   - (integer) the mask vector.
!        iz  - (integer) the stride in z.
!        x   - (integer) the source vector to be compressed.
!        ix  - (integer) the stride in x. 
!        iy  - (integer) the stride in y.                                                          
!                                                                       
!     OUTPUT ARGUMENTS -                                                
!                                                                       
!        y     - (integer) the compressed output vector.
!        count - (integer) the number of elements in y.                                                          
!                                                                       
!     CHANGE HISTORY -                                                  
!                                                                       
!        $Log: kmprsz.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.2   03/16/95 08:03:16   ejl
!PVCS    Cleaned up, Implicit none.
!PVCS    
!PVCS       Rev 1.1   01/04/95 21:55:52   llt
!PVCS    unicos changes (made by het)
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:42:44   pvcs
!PVCS    Original version.
!                                                                       
! ######################################################################
!                      
implicit none
!                                                                       
! ######################################################################
!
integer n, iz, ix, iy, count
integer z(*), x(*), y(*)
!                                                                       
! ######################################################################
!
integer imax, i, jx, jy
!
! ######################################################################
!
!
!
imax=iz*n
!
jx=1
jy=1
!
count=0
!
do i=1,imax,iz
!
   if(z(i)==0) then
      count=count+1
      y(jy)=x(jx)
      jy=jy+iy
   end if
!
   jx=jx+ix
!
end do
!
return
   end subroutine kmprsz
