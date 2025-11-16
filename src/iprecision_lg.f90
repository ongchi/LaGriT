! iprecision - Determine floating point precision
integer function iprecision()
  implicit none
  real :: a
  double precision :: b, c, x1, x2
  integer :: iprecision_test

  a = 1.0 / 3.0
  b = 1.0d+00 / 3.0d+00
  c = 1.0d+00 / 3.0d+00
  x1 = abs(a - c)
  x2 = abs(b - c)
  if (x1 == 0.0d+00) iprecision_test = 1
  if (x2 == 0.0d+00) iprecision_test = 2
  iprecision = iprecision_test

end function iprecision
