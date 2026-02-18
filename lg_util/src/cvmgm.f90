! cvmgm - Conditional value merge (minus/negative)
integer function cvmgm(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value < 0) then
    cvmgm = i1
  else
    cvmgm = i2
  end if

end function cvmgm
