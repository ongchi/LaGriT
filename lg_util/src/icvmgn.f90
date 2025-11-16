! icvmgn - Integer conditional value merge (non-zero)
integer function icvmgn(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value /= 0) then
    icvmgn = i1
  else
    icvmgn = i2
  end if

end function icvmgn
