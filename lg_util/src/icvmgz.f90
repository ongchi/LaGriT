! icvmgz - Integer conditional value merge (zero)
integer function icvmgz(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value == 0) then
    icvmgz = i1
  else
    icvmgz = i2
  end if

end function icvmgz
