logical function bit(nbitnum,ibitnum,iword)
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
!        $Log: bit.f,v $
!        Revision 2.00  2007/11/03 00:49:10  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.3   Mon Jan 25 17:59:26 1999   nnc
!PVCS    Changed declaration of ISHFT as EXTERNAL to INTRINSIC.
!PVCS    
!PVCS       Rev 1.2   Mon Jan 25 17:18:02 1999   nnc
!PVCS    Fixed error in external statement.
!PVCS    
!PVCS       Rev 1.1   Wed Jan 03 12:09:52 1996   het
!PVCS    Replace and() with iand()
!PVCS    
!PVCS       Rev 1.0   04/12/95 18:10:14   llt
!PVCS    Initial revision.
!
!#######################################################################
!
implicit none
!
integer nbitnum,ibitnum,iword(*)
integer itotal, ichange, j, jchange, nwd1, idum, imask
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
nwd1=ishft(iand(iword(jchange),imask(j)),-(j-1))
!
!     ******************************************************************
!     TEST THE CURRENT BIT SETTING. 
!         =  0 ==> FALSE
!        < > 0 ==> TRUE
!
if(nwd1==0) then
   bit=.false.
else
   bit=.true.
end if
!
!     ******************************************************************
!
return
   end function bit
