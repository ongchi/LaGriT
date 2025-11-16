*dk,iimin
      integer function iimin(n,ix,incx)
C
C ######################################################################
C
C        $Log: iimin.f,v $
C        Revision 2.00  2007/11/03 00:49:11  spchu
C        Import to CVS
C
CPVCS
CPVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
CPVCS    original version
C
C ######################################################################
C
      implicit none
      integer, intent(in) :: n, incx, ix(n)
      integer :: i, i1, icount

      iimin = -1
      i1 = ix(1)
      do i = 1, n, incx
         i1 = min(i1, ix(i))
      end do

      icount = 0
      do i = 1, n, incx
         icount = icount + 1
         if (ix(i) == i1) then
            iimin = icount
            exit
         endif
      end do

      end function iimin
