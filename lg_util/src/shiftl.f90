!dk,shiftl
integer function shiftl(num,i)
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
!        $Log: shiftl.f,v $
!        Revision 2.00  2007/11/03 00:49:13  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:45:46   pvcs
!PVCS    Original version.
!                                                                       
! ######################################################################
!                      
 implicit real*8 (a-h,o-z)
shiftl=ishft(num,i)
return
   end function shiftl
