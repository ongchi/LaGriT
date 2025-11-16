!dk,hpsortim
SUBROUTINE hpsortim(n,m,md,itemp,ia)
!
!
!#######################################################################
!
!     PURPOSE -
!
!     HPSORTIM ("HeaP SORT using an Integer M-fold key")
!     takes an integer N, and an integer array IA, with
!     first dimension length MD and second dimension length N and
!     shuffles the columns IA(1:MD,*) so that they are in 
!     lexicographic order up to the M'th key.  That is, if I<J
!     and IA(1,I) > IA(1,J), we interchange the I'th and the J'th
!     columns.  If IA(1,I) = IA(1,J), we then check if 
!     IA(2,I) > IA(2,J), in which case we again interchange the
!     columns.  Continuing on in this fashion, we consult the
!     elements IA(K,I) and IA(K,J) for K up to M if necessary
!     to break ties.  (M<=MD.) 
!
!     INPUT ARGUMENTS -
!
!        N - no. of columns to sort into ascending order.
!        M - maximum number of keys (rows) to consult for comparisons
!        MD- column length of array IA (M<=MD)
!        IA - integer array of MD-tuples to be reordered
!             into lexicographic ascending order up to the M'th key.
!        ITEMP - temp array of length MD.
!
!     OUTPUT ARGUMENTS -
!
!        IA - SORTED REAL*8 ARRAY.
!
!
!     CHANGE HISTORY -
!
!        $Log: hpsortim.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   22 Aug 2000 10:21:50   dcg
!PVCS    Initial revision.
!PVCS    
!PVCS       Rev 1.1   Tue Aug 31 15:06:56 1999   kuprat
!PVCS    During reordering, we now shuffle whole columns 
!PVCS    (i.e. if M<MD, we also copy entries beyond the M'th row).
!PVCS    
!PVCS       Rev 1.0   Tue Aug 24 21:57:04 1999   kuprat
!PVCS    Initial revision.
!
!#######################################################################
!
 
implicit none
 
INTEGER n,m,md
INTEGER ia(md,*)
INTEGER i,ir,j,l,k,k1
INTEGER itemp(md)
if (n<2) return
l=n/2+1
ir=n
10 continue
  if(l>1)then
    l=l-1
    do k=1,md
       itemp(k)=ia(k,l)
    end do
  else
    do k=1,md
       itemp(k)=ia(k,ir)
       ia(k,ir)=ia(k,1)
    end do
    ir=ir-1
    if(ir==1)then
        do k=1,md
           ia(k,1)=itemp(k)
        end do
      return
    end if
  end if
  i=l
  j=l+l
  if(j<=ir)then
    if(j<ir)then
!$$$            if(ra(j).lt.ra(j+1))j=j+1
      do k=1,m-1
        if(ia(k,j)<ia(k,j+1)) then
          j=j+1
          goto 30
        else if(ia(k,j)>ia(k,j+1)) then
          goto 30
        end if
      end do
      if(ia(m,j)<ia(m,j+1))j=j+1
    end if
30  continue
    do k=1,m-1
      if(itemp(k)<ia(k,j)) then
        do k1=1,md
           ia(k1,i)=ia(k1,j)
        end do
        i=j
        j=j+j
        goto 20
      else if(itemp(k)>ia(k,j)) then
        j=ir+1
        goto 20
      end if
    end do
    if(itemp(m)<ia(m,j)) then
      do k=1,md
         ia(k,i)=ia(k,j)
      end do
      i=j
      j=j+j
    else
      j=ir+1
    end if
    goto 20
  end if
20 continue
  do k=1,md
    ia(k,i)=itemp(k)
  end do
goto 10
   end subroutine hpsortim
