!dk,nxtilt
function nxtilt(array,number,target,stride)
!
! ######################################################################
!
!        $Log: nxtilt.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS
!PVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
!PVCS    original version
!
! ######################################################################
!
 implicit real*8 (a-h,o-z)
integer number,target,stride
integer array(number)
if(number<=0) then
   nxtilt=0
   goto 9999
end if
nxtilt=number+1
do i=1,number,stride
   if(array(i)<target) then
      nxtilt=i
      goto 9999
   end if
end do
goto 9999
9999 continue
return
   end function nxtilt
