!dk,hpsortip
SUBROUTINE hpsortip(n,ia,ascend,iprm)
!
!
!#######################################################################
!
!     PURPOSE -
!
!     HPSORTIP ("HeaP SORT an Integer array, producing a
!     Permutaion") takes an integer N, an integer array IA,
!     an integer array IPRM of length N, and a real
!     number ASCEND, and 
!     reorders IPRM so that IA(IPRM(1)),...,IA(IPRM(N))
!     is in ascending order if ASCEND is positive and
!     is in decreasing order if ASCEND is negative.
!
!     INPUT ARGUMENTS -
!
!        N - number of elements to be sorted.
!        IA - integer array of values which determine
!             how IPRM will be reordered.
!        IPRM - integer array to be reordered.  Strictly
!                        speaking, IPRM need not be a permutation of
!                        the integers {1,..,N}, but may be simply
!                        a mapping from {1,..,N} onto a set of N
!                        distinct positive integers.
!        ASCEND - real which controls whether we sort in ascending
!                 or descending order.
!
!     OUTPUT ARGUMENTS -
!
!        IPRM - reordered integer array.
!
!     CHANGE HISTORY -
!
!        $Log: hpsortip.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   22 Aug 2000 10:21:52   dcg
!PVCS    Initial revision.
!PVCS    
!PVCS       Rev 1.0   Tue Aug 24 21:57:50 1999   kuprat
!PVCS    Initial revision.
!
!#######################################################################
!
 
implicit none
 
INTEGER n,ia(*)
REAL*8 ascend
INTEGER i,ir,j,l,iprm(n),irra
if (n<2) return
l=n/2+1
ir=n
10 continue
  if(l>1)then
    l=l-1
    irra=iprm(l)
  else
    irra=iprm(ir)
    iprm(ir)=iprm(1)
    ir=ir-1
    if(ir==1)then
      iprm(1)=irra
      return
    end if
  end if
  i=l
  j=l+l
20 if(j<=ir)then
    if(j<ir)then
      if(ascend*ia(iprm(j))<ascend*ia(iprm(j+1)))j=j+1
    end if
    if(ascend*ia(irra)<ascend*ia(iprm(j)))then
      iprm(i)=iprm(j)
      i=j
      j=j+j
    else
      j=ir+1
    end if
  goto 20
  end if
  iprm(i)=irra
goto 10
   end subroutine hpsortip
