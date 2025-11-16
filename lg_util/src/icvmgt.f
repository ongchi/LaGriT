*dk,icvmgt
      integer function icvmgt(i1,i2,lmask)
C
C ######################################################################
C
C        $Log: icvmgt.f,v $
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
      integer, intent(in) :: i1, i2
      logical, intent(in) :: lmask

      if (lmask) then
         icvmgt = i1
      else
         icvmgt = i2
      endif

      end function icvmgt
