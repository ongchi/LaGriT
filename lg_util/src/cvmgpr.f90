! cvmgpr - Conditional value merge (positive, real comparison)
integer function cvmgpr(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value >= 0) then
    cvmgpr = i1
  else
    cvmgpr = i2
  end if

end function cvmgpr
