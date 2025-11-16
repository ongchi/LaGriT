! cvmgz - Conditional value merge (zero)
integer function cvmgz(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value == 0) then
    cvmgz = i1
  else
    cvmgz = i2
  end if

end function cvmgz
