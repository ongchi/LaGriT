!dk,shiftr
integer function shiftr(num,i)
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
!        $Log: shiftr.f,v $
!        Revision 2.00  2007/11/03 00:49:13  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:45:48   pvcs
!PVCS    Original version.
!                                                                       
! ######################################################################
!                      
 implicit real*8 (a-h,o-z)
shiftr= ishft(num,-i)
return
   end function shiftr
