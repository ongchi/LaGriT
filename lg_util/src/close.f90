subroutine close_lg (ifile)
!
!#######################################################################
!
!      PURPOSE -
!
!         THIS ROUTINE CLOSES A FILE THAT HAS BEEN PREVIOUSLY OPENDED
!            FOR I/O.
!
!      INPUT ARGUMENTS -
!
!         ifile    - THE FILE NAME.
!
!      OUTPUT ARGUMENTS -
!
!         NONE
!
!      CHANGE HISTORY -
!
!         $Log: close.f,v $
!         Revision 2.00  2007/11/03 00:49:10  spchu
!         Import to CVS
!
!PVCS    
!PVCS       Rev 1.6   Wed Feb 11 16:21:12 1998   dcg
!PVCS    don
!PVCS    skip inquire if file not opened
!PVCS
!PVCS       Rev 1.5   Wed Feb 11 15:39:32 1998   dcg
!PVCS    replace name of subroutine close with close_lg
!PVCS    this removes conflicts with some standard librarier
!PVCS
!PVCS       Rev 1.4   Thu Dec 18 10:24:16 1997   dcg
!PVCS    allow for 64 bit addresses by include rdwrt.h
!PVCS    use integer*8 version for 64 bit addresses
!PVCS    and link with 64 bit version of io package
!PVCS
!PVCS       Rev 1.3   10/20/95 13:07:52   dcg
!PVCS    check for existence
!PVCS
!PVCS       Rev 1.2   10/18/95 12:12:54   het
!PVCS    Add the dummy Fortran file/unit by preappending an F
!PVCS
!PVCS       Rev 1.1   07/14/95 10:11:32   het
!PVCS    Correct some errors for writing restart dumps
!PVCS
!PVCS       Rev 1.0   03/17/95 19:33:44   het
!PVCS    Original version converted from the CRAY
!
!#######################################################################
!
implicit none

!     Remove use of include rdwrt.h
!     The I/O library does not have entry points for handling 
!     type INTEGER*8 I/O control list specifiers
integer*4 iunitrw,iaddress,ierrrw

integer iflaga, ierror,lenmax
character ifile*(*)
logical opend
pointer (ipival, ival)
integer ival(10000000)
character*132 filename
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
   iunitrw=ival(1)
   inquire(unit=iunitrw,opened=opend,err=9999)
else
    inquire(file=ifile(1:lenmax), &
                opened=opend,number=iunitrw,err=9999)
end if
if(opend.eqv. .true.) then
   close(iunitrw)
   call cclose(iunitrw,ierrrw)
else
!        inquire(unit=iunitrw,name=filename,err=9999)
!        print *,"File already closed: ",iunitrw,filename
end if
9999 continue
return
end subroutine close_lg
!dk,closef
subroutine closef(ifile)
!
!#######################################################################
!
!      PURPOSE -
!
!         THIS ROUTINE CLOSES A FILE THAT HAS BEEN PREVIOUSLY OPENDED
!            FOR I/O.
!
!      INPUT ARGUMENTS -
!
!         ifile    - THE FILE NAME.
!
!      OUTPUT ARGUMENTS -
!
!         NONE
!
!      CHANGE HISTORY -
!
!         $Log: close.f,v $
!         Revision 2.00  2007/11/03 00:49:10  spchu
!         Import to CVS
!
!PVCS
!PVCS       Rev 1.1   07/14/95 10:11:32   het
!PVCS    Correct some errors for writing restart dumps
!PVCS
!PVCS       Rev 1.0   03/17/95 19:33:44   het
!PVCS    Original version converted from the CRAY
!
!#######################################################################
!
implicit none

!     The I/O library does not have entry points for handling
!     type INTEGER*8 I/O control list specifiers
integer*4 iunitrw,iaddress,ierrrw

integer iflaga, ierror,lenmax,iflag
character ifile*(*)
logical opend
pointer (ipival, ival)
integer ival(10000000)
integer icharlnf
character*132 filename, filename_fortran
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
   iunitrw=ival(1)
   inquire(unit=iunitrw,opened=opend,err=9999)
else
    filename_fortran='F' // ifile(1:lenmax)
    call fexist(filename_fortran,iflag)
    if (iflag==0) then
       print *,"File does not exist: ",filename_fortran
       go to 9999
    end if
    lenmax=lenmax+1
    inquire(file=filename_fortran(1:lenmax), &
                opened=opend,number=iunitrw,err=9999)
end if
if(opend.eqv. .true.) then
   close(iunitrw)
   call cclose(iunitrw,ierrrw)
else
   inquire(unit=iunitrw,name=filename)
   print *,"File already closed: ",iunitrw,filename
end if
9999 continue
return
end subroutine closef
