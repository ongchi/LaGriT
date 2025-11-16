*dk,icvmgtr
      double precision function icvmgtr(a1,a2,lmask)
C
C ######################################################################
C
C        $Log: icvmgtr.f,v $
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
      double precision, intent(in) :: a1, a2
      logical, intent(in) :: lmask

      if (lmask) then
        icvmgtr = a1
      else
        icvmgtr = a2
      endif

      end function icvmgtr
