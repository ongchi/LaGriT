! icvmgt - Integer conditional value merge function
integer function icvmgt(i1, i2, lmask)
  implicit none
  integer, intent(in) :: i1, i2
  logical, intent(in) :: lmask

  if (lmask) then
    icvmgt = i1
  else
    icvmgt = i2
  end if

end function icvmgt
