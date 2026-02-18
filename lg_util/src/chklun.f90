!dk,chklun
subroutine chklun(ifile, iflaga, ignore)
!       
!#######################################################################
!       
!      PURPOSE -
!       
!         THIS ROUTINE CHECK TO SEE IF A FILE HAS BEEN OPENED AND
!              ASSIGN A LOGICAL UNIT NUMBER.
!       
!      INPUT ARGUMENTS -
!       
!         ifile    - THE FILE NAME.
!         IGNORE   - IGNORE THIS FIELD.
!       
!      OUTPUT ARGUMENTS -       
!       
!         iflaga - INDICATES IF THE FILE (OR UNIT) NUMBER IS OPEN.
!                  = 0 ==> FILE IS NOT OPEN.
!                  = 1 ==> FILE IS OPEN.
!       
!      CHANGE HISTORY - 
!       
!         $Log: chklun.f,v $
!         Revision 2.00  2007/11/03 00:49:10  spchu
!         Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   03/17/95 19:33:42   het
!PVCS    Original version converted from the CRAY
!       
!#######################################################################
!       
character ifile*(*)
integer iflaga, ignore
logical opend

!     The I/O library does not have entry points for handling
!     type INTEGER*8 I/O control list specifiers
integer*4 iunit

pointer (ipival, ival(1))
integer icharlnf
!
!#######################################################################
!
iflaga=-1
ierror=-1
if(len(ifile)==0) then
   lenmax=0
!        *** IF THIS HAPPENS, THIS IS "PROABALY" AN INTEGER FIELD WITH A
!               WITH A CHARACTER CONTAINED WITHIN, MAKE AN ASSUMPTION.
else
   lenmax=icharlnf(ifile)
end if
if(lenmax==0) then
   ipival=loc(ifile)
   iunit=ival(1)
   inquire(unit=iunit,opened=opend,err=9999)
else
   inquire(file=ifile(1:lenmax),opened=opend,err=9999)
end if
if(opend.eqv..false.) then
   iflaga=0
else
   iflaga=1
end if
9999 continue
return
   end subroutine chklun
