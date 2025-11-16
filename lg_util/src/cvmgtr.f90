! cvmgtr - Conditional value merge (true, double precision)
double precision function cvmgtr(a1, a2, lmask)
  implicit none
  double precision, intent(in) :: a1, a2
  logical, intent(in) :: lmask

  if (lmask) then
    cvmgtr = a1
  else
    cvmgtr = a2
  end if

end function cvmgtr
