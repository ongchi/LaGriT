!dk,hpsorti
SUBROUTINE hpsorti(n,ia)
!
!
!#######################################################################
!
!     PURPOSE -
!
!     THIS IS BASED ON THE NUMERICAL RECIPES ROUTINE FOR SORTING
!     AN ARRAY USING THE 'HEAP SORT'.
!
!
!     INPUT ARGUMENTS -
!
!        n - NO. OF ELEMENTS TO SORT INTO ASCENDING ORDER.
!        ia - INTEGER ARRAY TO BE SORTED.
!
!
!
!     OUTPUT ARGUMENTS -
!
!        ia - SORTED INTEGER ARRAY.
!
!
!     CHANGE HISTORY -
!
!        $Log: hpsorti.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.0   22 Aug 2000 10:21:48   dcg
!PVCS    Initial revision.
!PVCS    
!PVCS       Rev 1.0   Tue Aug 24 21:58:34 1999   kuprat
!PVCS    Initial revision.
!
!#######################################################################
!
 
implicit none
 
INTEGER n
INTEGER ia(n)
INTEGER i,ir,j,l
INTEGER iia
if (n<2) return
l=n/2+1
ir=n
10 continue
  if(l>1)then
    l=l-1
    iia=ia(l)
  else
    iia=ia(ir)
    ia(ir)=ia(1)
    ir=ir-1
    if(ir==1)then
      ia(1)=iia
      return
    end if
  end if
  i=l
  j=l+l
20 if(j<=ir)then
    if(j<ir)then
      if(ia(j)<ia(j+1))j=j+1
    end if
    if(iia<ia(j))then
      ia(i)=ia(j)
      i=j
      j=j+j
    else
      j=ir+1
    end if
  goto 20
  end if
  ia(i)=iia
goto 10
   end subroutine hpsorti
