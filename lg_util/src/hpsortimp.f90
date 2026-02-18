!dk,hpsortimp
SUBROUTINE hpsortimp(n,m,md,ia,ascend,iprm)
!
!
!#######################################################################
!
!     PURPOSE -
!
!     HPSORTIMP ("HeaP SORT using an Integer M-fold key, generating
!     a Permution array") takes an integer N, an integer array IA, with
!     first dimension length MD and second dimension length N, an integer
!     permutation array IPRM of length N, and a real number ASCEND, and 
!     reorders IPRM so that IA(1,IPRM(1)),...,IA(1,IPRM(N))
!     is in ascending order if ASCEND is positive and
!     is in decreasing order if ASCEND is negative.  To 
!     break ties, we require also that IA(2,IPRM(*)) is
!     in ascending (descending) order, and so on, until
!     possibly the M'th key IA(M,*) is consulted
!     (i.e. ascending lexicographic order of M-tuples).
!     Of course this requires that M<=MD.
!
!     INPUT ARGUMENTS -
!
!        N - number of elements to be sorted.
!        M - we interpret array IA as M-tuples
!        MD- actual first dimension of array IA (M<=MD)
!        IA - integer array of values which determine
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
!        $Log: hpsortimp.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   08 May 2001 10:29:56   dcg
!PVCS    Initial revision.
!PVCS    
!PVCS       Rev 1.0   Tue Aug 24 21:57:34 1999   kuprat
!PVCS    Initial revision.
!
!#######################################################################
!
 
implicit none
 
INTEGER n,m,md
INTEGER ia(md,*)
REAL*8 ascend
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
20 if(j<=ir)then
    if(j<ir)then
      do k=1,m-1
        if(ascend*ia(k,iprm(j))<ascend*ia(k,iprm(j+1))) then
          j=j+1
          goto 30
        else if(ascend*ia(k,iprm(j))>ascend*ia(k,iprm(j+1)))then
          goto 30
        end if
      end do
      if(ascend*ia(m,iprm(j))<ascend*ia(m,iprm(j+1)))j=j+1
    end if
30  continue
    do k=1,m-1
      if(ascend*ia(k,irra)<ascend*ia(k,iprm(j))) then
        iprm(i)=iprm(j)
        i=j
        j=j+j
        goto 20
      else if(ascend*ia(k,irra)>ascend*ia(k,iprm(j))) then
        j=ir+1
        goto 20
      end if
    end do
    if(ascend*ia(m,irra)<ascend*ia(m,iprm(j))) then
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
   end subroutine hpsortimp
