!dk,heapexit
subroutine heapexit()
! 
!                                                                      
! ########################################################################
!                                                                       
!     PURPOSE -                                                         
!                                                                       
!        Stops code for memory management error.
!                                                                       
!     INPUT ARGUMENTS -                                                 
!                                                                       
!        NONE                                                           
!                                                                       
!     OUTPUT ARGUMENTS -                                                
!                                                                       
!        NONE                                                           
!                                                                       
!     CHANGE HISTORY -                                                  
!                                                                       
!        $Log: heapexit.f,v $
!        Revision 2.00  2007/11/03 00:49:10  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.1   05/30/95 13:46:04   ejl
!PVCS    Cleaned up, Implicit none.
!PVCS    
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:41:54   pvcs
!PVCS    Original version.
!                                                                       
! ########################################################################
!
implicit none
!
! ########################################################################
!                      
print *, 'Code stopping with a memory management error'
stop
!
   end subroutine heapexit
