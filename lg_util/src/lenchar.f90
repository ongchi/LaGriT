!dk,lenchar
function lenchar(iword)
!
!#######################################################################
!
!      PURPOSE -
!
!      FIND THE LENGTH OF A CHARACTER STRING BY SEARCHING BACKWARDS
!         UNTIL THE FIRST NON-BLANK CHARACTER IS FOUND.
!
!      INPUT ARGUMENTS -
!
!        iword    - A CHARACTER VARIABLE.
!
!      OUTPUT ARGUMENTS -
!
!        lenchar  - THE LENGTH OF THE CHARACTER STRING.
!
!      CHANGE HISTORY -
!
!        $Log: lenchar.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:42:58   pvcs
!PVCS    Original version.
!
!#######################################################################
!
character iword*(*)
!
!#######################################################################
!
do i=len(iword),1,-1
   if(iword(i:i)/=' ') then
      istart=1
      istop=i
      goto 100
   end if
end do
istart=1
istop=len(iword)
100 continue
lenchar=istop
goto 9999
9999 continue
return
   end function lenchar
