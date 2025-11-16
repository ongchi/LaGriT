!dk,lunget
function lunget(ifile)
!
!#######################################################################
!
!     PURPOSE -
!
!        THIS FUNCTION FINDS THE LOGICAL UNIT NUMBER ASSOCIATED
!           WITH THE FILE.
!
!     INPUT ARGUMENTS -
!
!        ifile    - THE FILE NAME TO INQUIRE ABOUT.
!
!     OUTPUT ARGUMENTS -
!
!        lunget - THE LOGICAL UNIT NUMBER CONNECTED TO "ifile".
!
!     CHANGE HISTORY -
!
!        $Log: lunget.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   11/10/94 12:42:58   pvcs
!PVCS    Original version.
!
!#######################################################################
!
character ifile*(*)
logical opend

!     The I/O library does not have entry points for handling
!     type INTEGER*8 I/O control list specifiers
integer*4 kunit,lunget,jfile

!
!#######################################################################
!
if(len(ifile)==0) then
   lenmax=8
!        *** IF THIS HAPPENS, THIS IS "PROABALY" AN INTEGER FIELD WITH A
!               WITH A CHARACTER CONTAINED WITHIN, MAKE AN ASSUMPTION.
!********jfile=int(ifile)
   read(ifile,'(i8)') jfile
   inquire(unit=jfile, &
                number=kunit,opened=opend,err=9999)
   if(opend.eqv..true.) then
      lunget=kunit
   else
      lunget=-1
      inquire(file=ifile(1:lenmax), &
                   opened=opend,number=kunit,err=9999)
      if(opend.eqv..true.) then
         lunget=kunit
      end if
   end if
else
   lenmax=icharlnf(ifile)
   lunget=-1
   inquire(file=ifile(1:lenmax),opened=opend,err=9999)
   if(opend.eqv..true.) then
      inquire(file=ifile(1:lenmax),number=kunit)
      lunget=kunit
   end if
end if
if(ifile(1:3)=='tty') lunget=6
goto 9999
9999 continue
return
   end function lunget
