! cvmgmr - Conditional value merge (minus/negative, real comparison)
integer function cvmgmr(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value < 0) then
    cvmgmr = i1
  else
    cvmgmr = i2
  end if

end function cvmgmr
