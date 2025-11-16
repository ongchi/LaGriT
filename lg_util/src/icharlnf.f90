!
integer function icharlnf(iword)
!
!
!#######################################################################
!
!      PURPOSE -
!
!      Find the Length of a Character String by Searching Forward
!         until the first Blank Character is found after the fist
!         non-Blank Character.
!
!      INPUT ARGUMENTS -
!
!        iword    - (character) A Character String.
!
!      OUTPUT ARGUMENTS -
!
!        icharlnf  - (integer) The length of the Character String.
!
!      CHANGE HISTORY -
!
!        $Log: icharlnf.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.1   02/23/95 13:09:26   ejl
!PVCS     
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:42:08   pvcs
!PVCS    Original version.
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
lenmax=len(iword)
!
istop=0
!
!.... Find the first Non-Blank Character.
!
do while ((istop < lenmax) .and. &
               (iword(istop+1:istop+1) == ' '))
!
   istop=istop+1
!
end do
!
!.... Now find the first Blank or Null Character.
!
do while ((istop < lenmax) .and. &
               (iword(istop+1:istop+1) /= char(0)) .and. &
               (iword(istop+1:istop+1) /= ' '))
!
   istop=istop+1
!
end do
!
!.... Number of characters in iword.
!
icharlnf=istop
!
return
   end function icharlnf
