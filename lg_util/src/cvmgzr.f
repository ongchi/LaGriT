*dk,cvmgzr
      integer function cvmgzr(i1,i2,value)
C
C ######################################################################
C
C        $Log: cvmgzr.f,v $
C        Revision 2.00  2007/11/03 00:49:10  spchu
C        Import to CVS
C
CPVCS
CPVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
CPVCS    original version
C
C ######################################################################
C
      implicit none
      integer, intent(in) :: i1, i2, value

      if (value == 0) then
         cvmgzr = i1
      else
         cvmgzr = i2
      endif

      end function cvmgzr
