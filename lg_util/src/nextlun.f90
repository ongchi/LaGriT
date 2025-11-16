!dk,nextlun
function nextlun()
!
!#######################################################################
!
!     PURPOSE -
!
!        THIS FUNCTION PICKS THE NEXT FREE LOGICAL UNIT NUMBER.
!
!     INPUT ARGUMENTS -
!
!        NONE
!
!     OUTPUT ARGUMENTS -
!
!        nextlun - THE NEXT LOGICAL UNIT NUMBER.
!
!     CHANGE HISTORY -
!
!        $Log: nextlun.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:43:44   pvcs
!PVCS    Original version.
!
!#######################################################################
!

!     The I/O library does not have entry points for handling
!     type INTEGER*8 I/O control list specifiers
integer*4 iunit,kunit,nextlun
logical opend
!
!#######################################################################
!
do 100 iunit=1,99
   if(iunit>=5.and.iunit<=7) goto 100
   inquire(iunit,opened=opend,err=100)
   if(opend.eqv..false.) then
      kunit=iunit
      goto 200
   end if
100 continue
200 continue
nextlun=kunit
goto 9999
9999 continue
return
   end function nextlun
