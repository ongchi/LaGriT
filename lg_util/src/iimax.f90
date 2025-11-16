! iimax - Find index of maximum value in integer array
integer function iimax(n, ix, incx)
  implicit none
  integer, intent(in) :: n, incx, ix(n)
  integer :: i, i1, icount

  iimax = -1
  i1 = ix(1)
  do i = 1, n, incx
    i1 = max(i1, ix(i))
  end do

  icount = 0
  do i = 1, n, incx
    icount = icount + 1
    if (ix(i) == i1) then
      iimax = icount
      exit
    end if
  end do

end function iimax
