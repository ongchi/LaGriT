! cvmgt - Conditional value merge function
integer function cvmgt(i1, i2, lmask)
!
! ######################################################################
!
!        $Log: cvmgt.f,v $
!        Revision 2.00  2007/11/03 00:49:10  spchu
!        Import to CVS
!
! PVCS
! PVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
! PVCS    original version
!
! ######################################################################
!
  implicit none
  integer, intent(in) :: i1, i2
  logical, intent(in) :: lmask

  if (lmask) then
    cvmgt = i1
  else
    cvmgt = i2
  end if

end function cvmgt
