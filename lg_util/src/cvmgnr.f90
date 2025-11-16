! cvmgnr - Conditional value merge (non-zero, real comparison)
integer function cvmgnr(i1, i2, value)
  implicit none
  integer, intent(in) :: i1, i2, value

  if (value /= 0) then
    cvmgnr = i1
  else
    cvmgnr = i2
  end if

end function cvmgnr
