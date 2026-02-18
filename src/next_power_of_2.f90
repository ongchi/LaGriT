! next_power_of_2 - Calculate the power of 2 >= ii
! i.e.  2**ceiling(log2(ii))
integer function next_power_of_2(ii)
  implicit none
  integer, intent(in) :: ii
  real :: base_2_log_ii
  integer :: ii_log_2

  if (ii <= 1) then
    next_power_of_2 = 1
  else
    ! So ii > 1
    base_2_log_ii = log(real(ii)) / log(2.0)
    ii_log_2 = int(base_2_log_ii)
    ! *** truncate base_2_log_ii
    if (2**ii_log_2 >= ii) then
      next_power_of_2 = 2**ii_log_2
    else
      next_power_of_2 = 2**(ii_log_2 + 1)
    end if
  end if

end function next_power_of_2
