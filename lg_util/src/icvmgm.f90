! icvmgm - Integer conditional value merge (minus/negative)
integer function icvmgm(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value < 0) then
    icvmgm = i1
  else
    icvmgm = i2
  end if

end function icvmgm
