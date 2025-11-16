!dk,scopyii
subroutine scopyii(n,isource,isource_stride,isink,isink_stride)
!
! #####################################################################
!
!     PURPOSE -
!
!        THIS ROUTINE WRITES A DUMP FILE FOR CHAD.
!
!     INPUT ARGUMENTS -
!
!        n              - NUMBER OF ELEMENTS TO COPY.
!        isource        - INTEGER SOURCE ARRAY
!        isource_stride - INTEGER SOURCE ARRAY STRIDE
!        isink_stride   - INTEGER SINK ARRAY STRIDE
!
!     OUTPUT ARGUMENTS  -
!
!        isink          - INTEGER SINK ARRAY
!
!
!     CHANGE HISTORY -
!
!        $Log: scopyii.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   01/17/95 16:42:04   pvcs
!PVCS    Original Version
!
! ######################################################################
!
implicit real*8 (a-h,o-z)
!
! ######################################################################
!
dimension isource(n), isink(n)
!
! ######################################################################
!
i1=1
i2=1
do i=1,n
   isink(i2)=isource(i1)
   i1=i1+isource_stride
   i2=i2+isink_stride
end do
goto 9999
9999 continue
return
   end subroutine scopyii
