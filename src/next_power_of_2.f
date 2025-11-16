*dk next_power_of_2
      integer function next_power_of_2(ii)
C
C#######################################################################
C        $Log: next_power_of_2.f,v $
C        Revision 2.00  2007/11/05 19:46:02  spchu
C        Import to CVS
C
CPVCS
CPVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
CPVCS    original version
C
C#######################################################################
C
C  Calculate the power of 2 >= ii
C  i.e.  2**ceiling(log2(ii))
C
      implicit none
      integer, intent(in) :: ii
      real :: base_2_log_ii
      integer :: ii_log_2

      if (ii <= 1) then
         next_power_of_2 = 1
      else
C       So ii > 1
        base_2_log_ii = log(real(ii)) / log(2.0)
        ii_log_2 = int(base_2_log_ii)
C          *** truncate base_2_log_ii
        if (2**ii_log_2 >= ii) then
           next_power_of_2 = 2**ii_log_2
        else
           next_power_of_2 = 2**(ii_log_2 + 1)
        endif
      endif

      end function next_power_of_2
