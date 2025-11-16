! cvmgn - Conditional value merge (non-zero)
integer function cvmgn(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value /= 0) then
    cvmgn = i1
  else
    cvmgn = i2
  end if

end function cvmgn
