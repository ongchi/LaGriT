!dk,fexist
subroutine fexist(ifile, iflaga)
!       
!#######################################################################
!       
!      PURPOSE -
!       
!         THIS ROUTINE CHECKS THE EXISTENCE OF A FILE.
!       
!      INPUT ARGUMENTS -
!       
!         ifile    - THE FILE NAME.
!       
!      OUTPUT ARGUMENTS -       
!       
!         iflaga - = 0 ==> FILE DOES NOT EXIST.
!                  = 1 ==> FILE DOES EXIST.
!       
!      CHANGE HISTORY - 
!       
!         $Log: fexist.f,v $
!         Revision 2.00  2007/11/03 00:49:10  spchu
!         Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   03/17/95 19:34:50   het
!PVCS    Original version converted from the CRAY
!       
!#######################################################################
!       

character ifile*(*)
logical iexist
pointer (ipival, ival(1))

!     The I/O library does not have entry points for handling
!     type INTEGER*8 I/O control list specifiers
integer*4 iunit
       
!
!#######################################################################
!
iflaga=0
ierror=-1
if(len(ifile)==0) then
   lenmax=0
!        *** IF THIS HAPPENS, THIS IS "PROABALY" AN INTEGER FIELD WITH A
!               WITH A CHARACTER CONTAINED WITHIN, MAKE AN ASSUMPTION.
else
   lenmax=icharlnf(ifile)
end if
iexist=.false.
if(lenmax==0) then
   ipival=loc(ifile)
   iunit=ival(1)
   inquire(unit=iunit,exist=iexist,err=9999)
else
   inquire(file=ifile(1:lenmax),exist=iexist,err=9999)
end if
if(iexist .eqv. .true.) then
   iflaga=1
else if(iexist .eqv. .false.) then
   iflaga=0
end if
goto 9999
9999 continue
return
   end subroutine fexist
