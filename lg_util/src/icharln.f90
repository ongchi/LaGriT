!dk,icharln
integer function icharln(iword)
!
!
!#######################################################################
!
!      PURPOSE -
!
!      Find the Length of a Character String by Searching Forward
!         until the first Blank or Null Character is found..
!
!      INPUT ARGUMENTS -
!
!        iword    - (character) A Character String.
!
!      OUTPUT ARGUMENTS -
!
!        icharln  - (integer) The length of the Character String.
!
!      CHANGE HISTORY -
!
!        $Log: icharln.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   05/31/95 10:45:52   ejl
!PVCS    New function to return lwength of left-justified character string.
!PVCS    
!
!#######################################################################
!
implicit none
!
!#######################################################################
!
character*(*) iword
!
!#######################################################################
!
!     LOCAL VARIABLE DEFINITION
!
integer lenmax, istop
!
!#######################################################################
!
!
!
lenmax = len(iword)
!
istop=0
!
!.... Find the first Blank or Null Character.
!
do while ((istop < lenmax) .and. &
               (iword(istop+1:istop+1) /= char(0)) .and. &
               (iword(istop+1:istop+1) /= ' '))
!
   istop = istop+1
!
end do
!
!.... Number of characters in iword.
!
icharln = istop
!
return
   end function icharln
