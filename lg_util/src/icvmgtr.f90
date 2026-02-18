! icvmgtr - Integer conditional value merge (true, double precision)
double precision function icvmgtr(a1, a2, lmask)
  implicit none
  double precision, intent(in) :: a1, a2
  logical, intent(in) :: lmask

  if (lmask) then
    icvmgtr = a1
  else
    icvmgtr = a2
  end if

end function icvmgtr
