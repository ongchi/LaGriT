
subroutine hassign(iunit,ifile,ibuffer)

!
!#######################################################################
!
!     PURPOSE -
!
!        THIS ROUTINE ASSIGNS (OPENS) A FILE.
!
!     INPUT ARGUMENTS -
!
!        iunit    - UNIT NUMBER FLAG:  =  -1 ==> ROUTINE SHOULD PICK AN
!                                                UNUSED UNIT NUMBER.
!                                      <> -1 ==> TEST THIS SPECIFIC UNIT
!                                                   TO SEE IF IT HAS
!                                                   BEEN USED.
!        ifile    - FILE NAME TO BE ASSIGNED (OPENED).
!        ibuffer  - BUFFER SIZE (=0 ALMOST ALWAYS).
!                 - allow ibuffer to hold ierror on return
!                   since this is how this argument is used
!                   for most code using this subroutine
!                   ibuffer seems to be ignored most times
!
!     OUTPUT ARGUMENTS -
!
!        iunit - THE UNIT NUMBER ASSIGED FOR THIS FILE.
!
!     CHANGE HISTORY -
!
!        $Log: hassign.f,v $
!        Revision 2.00  2007/11/03 00:49:10  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.1   01/04/95 21:55:08   llt
!PVCS    unicos changes (made by het)
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:41:52   pvcs
!PVCS    Original version.
!
!#######################################################################
!
! args
character ifile*(*)
integer iunit
integer ibuffer

!     The I/O library does not have entry points for handling
!     type INTEGER*8 I/O control list specifiers
!     but keep arguments as integer

integer*4 iunit4, kunit
integer ierror
integer lenmax
integer icharlnf

logical opend
!
!#######################################################################
! begin

ierror=-1
iunit4 = iunit

if(len(ifile)==0) then
   lenmax=8
!        *** IF THIS HAPPENS, THIS IS "PROABALY" AN INTEGER FIELD WITH A
!               WITH A CHARACTER CONTAINED WITHIN, MAKE AN ASSUMPTION.
else
   lenmax=icharlnf(ifile)
end if
if(iunit4<=0) then
   ierror=0
   iunit4=nextlun()
   open(unit=iunit4,file=ifile(1:lenmax))
else
   inquire(iunit4,opened=opend,err=9999)
   if(opend.eqv..false.) then
      ierror=0
      kunit=iunit4
      open(unit=iunit4,file=ifile(1:lenmax))
   else
      ierror=0
      iunit4=nextlun()
      open(unit=iunit4,file=ifile(1:lenmax))
   end if
end if
goto 9999
9999 continue

!     iunit is returned with the file number
!     ibuffer is returned with error value

iunit = iunit4
ibuffer = ierror
return
   end subroutine hassign
