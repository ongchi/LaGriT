!dk,compl
integer function compl(num)
!
! #####################################################################
!
!     PURPOSE -
!
!        None
!
!     INPUT ARGUMENTS -
!
!        None
!
!     OUTPUT ARGUMENTS -
!
!        None
!
!     CHANGE HISTORY -
!
!        $Log: compl.f,v $
!        Revision 2.00  2007/11/03 00:49:10  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.2   10 Aug 2005 10:41:30   dcg
!PVCS    replace .not. with  call to function not
!PVCS
!PVCS       Rev 1.1   01/04/95 21:54:16   llt
!PVCS    unicos changes (made by het)
!PVCS
!PVCS       Rev 1.0   11/10/94 12:40:58   pvcs
!PVCS    Original version.
!
! ######################################################################
!
 implicit real*8 (a-h,o-z)
compl=not(num)
return
   end function compl
