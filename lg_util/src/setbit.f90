subroutine setbit(nbitnum,ibitnum,iword,istate)
!
!#######################################################################
!
!      PURPOSE -
!
!         THIS ROUTINE SETS THE SPECIFIED BIT TO THE INPUT STATE.
!            THE BITS ARE ARRANGED FROM LEAST-SIGNIFICANT-BIT (0 WHICH
!            IS ON THE RIGHT) TO THE MOST-SIGNIFICANT-BIT (nbitnum
!            WHICH IS ON THE LEFT).
!
!      INPUT ARGUMENTS -
!
!         nbitnum - TOTAL NUMBER OF BITS IN THE INPUT WORD. THIS
!                      CAN BE ANY LENGTH.
!         ibitnum - THE BIT THAT IS TO BE CHANGED.
!         iword() - THE STARTING LOCATION OF THE MOST-SIGNIFICANT-BIT.
!         istate  - THE STATE OF THE BIT TO BE SET.
!
!      OUTPUT ARGUMENTS -
!
!         iword() - THE STARTING LOCATION OF THE MOST-SIGNIFICANT-BIT.
!
!      CHANGE HISTORY -
!
!        $Log: setbit.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.6   Mon Jan 25 17:58:58 1999   nnc
!PVCS    Changed declaration of ISHFT as EXTERNAL to INTRINSIC.
!PVCS    
!PVCS       Rev 1.5   Mon Jan 25 17:17:48 1999   nnc
!PVCS    Fixed error in external statement.
!PVCS    
!PVCS       Rev 1.4   Tue Jun 18 09:21:14 1996   dcg
!PVCS    use ior and iand in place of or and and
!PVCS
!PVCS       Rev 1.3   12/05/95 08:32:24   het
!PVCS    Make UNICOS changes
!PVCS
!PVCS       Rev 1.2   11/21/95 11:52:20   dcg
!PVCS    add arbitrary length or, and functions
!PVCS
!PVCS       Rev 1.0   04/12/95 18:10:16   llt
!PVCS    Initial revision.
!
!#######################################################################
!
implicit none
!
integer nbitnum,ibitnum,istate,iword(*)
integer itotal, ichange, j, jchange, nwd1, jstate, idum, imask
integer maskv1, maskv2
integer ishft
intrinsic ishft
!
!#######################################################################
!
!     DEFINE A STATEMENT FUNCTION FOR CREATING A MASK BY SHIFTING
!        BY A SPECIFIED NUMBER OF BITS.
!
imask(idum)=2**(idum-1)
!
!#######################################################################
!
!     *** CALCULATE THE TOTAL NUMBER OF 32-BIT ELEMENTS IN THE INPUT
!            VECTOR.
itotal=1+(nbitnum-1)/32
!
!     *** CALCULATE THE INDEX NUMBER OF THE 32-BIT ELEMENT TO CHANGE.
ichange=1+(ibitnum-1)/32
!
!     *** CALCULATE THE BIT NUMBER OF THE 32-BIT ELEMENT TO CHANGE.
j=ibitnum-32*(ichange-1)+1
!
!     *** CALCULATE THE INDEX NUMBER OF THE INPUT VECTOR TO CHANGE.
jchange=itotal-ichange+1
!
!     *** FIND THE CURRENT VALUE OF THE BIT IN THE INPUT VECTOR.
maskv1=iword(jchange)
maskv2=imask(j)
nwd1=ishft(iand(maskv1,maskv2),-(j-1))
!
!     ******************************************************************
!     SET THE BIT TO THE INPUT STATE BY DOING A TRUTH TABLE WITH THE
!        CURRENT BIT SETTING AND THE STATE VALUE.
!
!        CURRENT BIT  STATE VALUE  OUTPUT BIT
!        ***********  ***********  **********
!                  0            1           1
!                  1            0           0
!                  1            1           1
!                  0            0           0
!
if(nwd1==0.and.istate/=0) then
   jstate=ishft(1,(j-1))
   iword(jchange)=ior(iword(jchange),jstate)
else if(nwd1==1.and.istate==0) then
   jstate=not(ishft(1,(j-1)))
   iword(jchange)=iand(iword(jchange),jstate)
else if(nwd1==1.and.istate/=0) then
else if(nwd1==0.and.istate==0) then
end if
!
!     ******************************************************************
!
goto 9999
9999 continue
return
end subroutine setbit
subroutine fpor(numbits,x,y,z)
!
! #######################################################
!
!   PURPOSE
!      perform a logical or operation on an
!      arbitrary number of bits
!
!   INPUT ARGUMENTS
!      numbits -- number of bits to 'or'
!      x          first argument of 'or'
!      y          second argument of 'or'
!
!   OUTPUT ARGUMENTS
!      z    result of 'or' on x or y
!
! #######################################################
implicit none
logical bit
real*8 x(*),y(*),z(*)
integer i1,i2,i3,i
integer numbits
do i=1,numbits
i1=0
i2=0
if(bit(numbits,i-1,x) ) i1=1
if(bit(numbits,i-1,y) ) i2=1
i3 = ior(i1,i2)

! 32 bit compiler warning
! Warning: Type mismatch in argument 'iword' 
! at (1); passed REAL(8) to INTEGER(4)
! 64 bit; passed REAL(8) to INTEGER(8)
call setbit(numbits,i-1,z,i3)
end do
return
end subroutine fpor
subroutine fpand(numbits,x,y,z)
!
! #######################################################
!
!   PURPOSE
!      perfandm a logical and operation on an
!      arbitrary number of bits
!
!   INPUT ARGUMENTS
!      numbits -- number of bits to 'and'
!      x          first argument of 'and'
!      y          second argument of 'and'
!
!   OUTPUT ARGUMENTS
!      z    result of 'and' on x and y
!
! #######################################################
implicit none
logical bit
real*8 x(*),y(*),z(*)
integer i1,i2,i3,i
integer numbits
do i=1,numbits
i1=0
i2=0
if(bit(numbits,i-1,x) ) i1=1
if(bit(numbits,i-1,y) ) i2=1
i3 = iand(i1,i2)

! 32 bit compiler warning
! Warning: Type mismatch in argument 'iword' at (1); 
! passed REAL(8) to INTEGER(4)
! 64 bit; passed REAL(8) to INTEGER(8)
call setbit(numbits,i-1,z,i3)
end do
return
   end subroutine fpand
