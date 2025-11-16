integer function icharlnb(iword)
!
!
!#######################################################################
!
!      PURPOSE -
!
!      Find the Length of a Character String by Searching Backwards
!         until the first Non-Blank and Non-Null Character is found.
!
!      INPUT ARGUMENTS -
!
!        iword    - (character) A Character String.
!
!      OUTPUT ARGUMENTS -
!
!        icharlnb  - (integer) The Length of the Character String.
!
!      CHANGE HISTORY -
!
!        $Log: icharlnb.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.1   02/23/95 13:09:22   ejl
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
integer istop
!
!#######################################################################
!
!
!
istop=len(iword)
!
!.... Find the first Non-Blank and Non-Null Character.
!
do while ((istop > 0) .and. &
               ((iword(istop:istop) == ' ') .or. &
                (iword(istop:istop) == char(0))))
!
   istop=istop-1
!
end do

icharlnb=istop
!
return
   end function icharlnb
