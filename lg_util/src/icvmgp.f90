! icvmgp - Integer conditional value merge (positive)
integer function icvmgp(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value >= 0) then
    icvmgp = i1
  else
    icvmgp = i2
  end if

end function icvmgp
