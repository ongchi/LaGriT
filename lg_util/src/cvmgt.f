*dk,cvmgt
      integer function cvmgt(i1,i2,lmask)
C
C ######################################################################
C
C        $Log: cvmgt.f,v $
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
      integer, intent(in) :: i1, i2
      logical, intent(in) :: lmask

      if (lmask) then
         cvmgt = i1
      else
         cvmgt = i2
      endif

      end function cvmgt
