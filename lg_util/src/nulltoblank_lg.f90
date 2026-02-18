subroutine nulltoblank_lg (iword,length)
!
!
!#######################################################################
!
!      PURPOSE -
!
!      Change terminating null to a blank.
!
!      INPUT ARGUMENTS -
!
!        iword    - (character) A Character String.
!        length   - length of iword
!
!      OUTPUT ARGUMENTS -
!
!        iword     - (character) Modified character variable with null
!                      replaced by a blank
!
!      CHANGE HISTORY -
!        $Log: nulltoblank_lg.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   Wed Apr 12 15:08:12 2000   dcg
!PVCS    Initial revision.
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
integer i,length
!
!#######################################################################
!
!
!.... See if a null exists
!
do i=length,1,-1
   if(iword(i:i)==char(0)) then
       iword(i:i)=' '
   end if
end do
return
   end subroutine nulltoblank_lg
