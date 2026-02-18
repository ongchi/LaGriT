!dk,ssum
function ssum(n,x,ix)
!
! ######################################################################
!
!        $Log: ssum.f,v $
!        Revision 2.00  2007/11/03 00:49:13  spchu
!        Import to CVS
!
!PVCS
!PVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
!PVCS    original version
!
! ######################################################################
!
 implicit real*8 (a-h,o-z)
dimension x(n)
ssum=0.0
do i=1,n,ix
   ssum=ssum+x(i)
end do
goto 9999
9999 continue
return
   end function ssum
