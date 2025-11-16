! cvmgzr - Conditional value merge (zero, real comparison)
integer function cvmgzr(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value == 0) then
    cvmgzr = i1
  else
    cvmgzr = i2
  end if

end function cvmgzr
