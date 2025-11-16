! cvmgp - Conditional value merge (positive)
integer function cvmgp(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value >= 0) then
    cvmgp = i1
  else
    cvmgp = i2
  end if

end function cvmgp
