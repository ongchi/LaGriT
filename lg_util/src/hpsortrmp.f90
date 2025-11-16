!dk,hpsortrmp
SUBROUTINE hpsortrmp(n,m,md,a,ascend,iprm)
!
!
!#######################################################################
!
!     PURPOSE -
!
!     HPSORTRMP ("HeaP SORT using a Real*8 M-fold key, generating
!     a Permution array") takes an integer N, a real*8 array A, with
!     first dimension length MD and second dimension length N, an integer
!     permutation array IPRM of length N, and a real*8 number ASCEND, and 
!     reorders IPRM so that A(1,IPRM(1)),...,A(1,IPRM(N))
!     is in ascending order if ASCEND is positive and
!     is in decreasing order if ASCEND is negative.  To 
!     break ties, we require also that A(2,IPRM(*)) is
!     in ascending (descending) order, and so on, until
!     possibly the M'th key A(M,*) is consulted
!     (i.e. ascending lexicographic order of M-tuples).
!     Of course this requires that M<=MD.
!
!     INPUT ARGUMENTS -
!
!        N - number of elements to be sorted.
!        M - we interpret array A as M-tuples
!        MD- actual first dimension of array A (M<=MD)
!        A - real*8 array of values which determine
!             how IPRM will be reordered, it has 'depth' M for 
!             purposes of tie-breaking
!        IPRM - integer array to be reordered.  Strictly
!             speaking, IPRM need not be a permutation of
!             the integers {1,..,N}, but may be simply
!             a mapping from {1,..,N} onto a set of N
!             distinct positive integers.
!        ASCEND - real*8 which controls whether we sort in ascending
!                 or descending order.
!
!     OUTPUT ARGUMENTS -
!
!        IPRM - reordered integer array.
!
!     CHANGE HISTORY -
!
!        $Log: hpsortrmp.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   22 Aug 2000 10:21:54   dcg
!PVCS    Initial revision.
!PVCS    
!PVCS       Rev 1.0   Mon Nov 15 13:22:40 1999   kuprat
!PVCS    Initial revision.
!
!#######################################################################
!
 
implicit none
 
INTEGER n,m,md
REAL*8 ascend,a(md,*)
INTEGER i,ir,j,l,iprm(n),irra,k
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
  if(j<=ir)then
    if(j<ir)then
      do k=1,m-1
        if(ascend*a(k,iprm(j))<ascend*a(k,iprm(j+1))) then
          j=j+1
          goto 30
        else if(ascend*a(k,iprm(j))>ascend*a(k,iprm(j+1)))then
          goto 30
        end if
      end do
      if(ascend*a(m,iprm(j))<ascend*a(m,iprm(j+1)))j=j+1
    end if
30  continue
    do k=1,m-1
      if(ascend*a(k,irra)<ascend*a(k,iprm(j))) then
        iprm(i)=iprm(j)
        i=j
        j=j+j
        goto 20
      else if(ascend*a(k,irra)>ascend*a(k,iprm(j))) then
        j=ir+1
        goto 20
      end if
    end do
    if(ascend*a(m,irra)<ascend*a(m,iprm(j))) then
      iprm(i)=iprm(j)
      i=j
      j=j+j
    else
      j=ir+1
    end if
    goto 20
  end if
20 continue
  iprm(i)=irra
goto 10
   end subroutine hpsortrmp
