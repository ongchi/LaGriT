integer function ismax(N,SX,INCX)
!
! ######################################################################
!
!        $Log: ismax.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS
!PVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
!PVCS    original version
!
! ######################################################################
!
implicit none
!  this function operates as double precision on short word machines
!***BEGIN PROLOGUE  ismax
!***DATE WRITTEN   790614   (YYMMDD)
!***REVISION DATE  860401   (YYMMDD)
!***CATEGORY NO.  D1A2
!***KEYWORDS  VECTOR,MAXIMUM,INDEX
!***AUTHOR  KAHANER, D. K., LOS ALAMOS NATIONAL LABORATORY
!***PURPOSE  Find the smallest index of a maximum element of a vector.
!***DESCRIPTION
!
!   This function finds the smallest index of a maximum element of a
!   real array SX whose N elements are stored sequentially with
!   spacing INCX >= 1.  If N <= 0, the value zero is returned.
!   Thus, if I = ismax(N,SX,1), then SX(I) is an element of array SX
!   of maximum value.
!
!   Description of Parameters
!
!    --Input--
!        N  number of elements in input vector
!       SX  single precision vector with N elements
!     INCX  storage spacing between elements of SX
!
!    --Output--
!    ismax   smallest index (zero if N .LE. 0)
!
!***REFERENCES  (NONE)
!***ROUTINES CALLED  (NONE)
!***END PROLOGUE  ismax
INTEGER I,INCX,IX,N
real*8 SX(N),SMAX
!***FIRST EXECUTABLE STATEMENT  ismax
ismax = 0
IF( N < 1 ) RETURN
ismax = 1
IF(N==1)RETURN
IF(INCX==1)GO TO 20
!
!        CODE FOR INCREMENT NOT EQUAL TO 1
!
IX = 1
SMAX = (SX(1))
IX = IX + INCX
DO 10 I = 2,N
   IF((SX(IX))<=SMAX) GO TO 5
   ismax = I
   SMAX = (SX(IX))
5  IX = IX + INCX
10 CONTINUE
RETURN
!
!        CODE FOR INCREMENT EQUAL TO 1
!
20 SMAX = (SX(1))
DO 30 I = 2,N
   IF((SX(I))<=SMAX) GO TO 30
   ismax = I
   SMAX = (SX(I))
30 CONTINUE
RETURN
end function ismax
!
integer function ismaxi(N,SX,INCX)
implicit none
!  this function operates as double precision on short word machines
!***BEGIN PROLOGUE  ismax
!***DATE WRITTEN   790614   (YYMMDD)
!***REVISION DATE  860401   (YYMMDD)
!***CATEGORY NO.  D1A2
!***KEYWORDS  VECTOR,MAXIMUM,INDEX
!***AUTHOR  KAHANER, D. K., LOS ALAMOS NATIONAL LABORATORY
!***PURPOSE  Find the smallest index of a maximum element of a vector.
!***DESCRIPTION
!
!   This function finds the smallest index of a maximum element of a
!   real array SX whose N elements are stored sequentially with
!   spacing INCX >= 1.  If N <= 0, the value zero is returned.
!   Thus, if I = ismax(N,SX,1), then SX(I) is an element of array SX
!   of maximum value.
!
!   Description of Parameters
!
!    --Input--
!        N  number of elements in input vector
!       SX  single precision vector with N elements
!     INCX  storage spacing between elements of SX
!
!    --Output--
!    ismax   smallest index (zero if N .LE. 0)
!
!***REFERENCES  (NONE)
!***ROUTINES CALLED  (NONE)
!***END PROLOGUE  ismax
INTEGER I,INCX,IX,N
integer SX(N),SMAX
!***FIRST EXECUTABLE STATEMENT  ismax
ismaxi = 0
IF( N < 1 ) RETURN
ismaxi = 1
IF(N==1)RETURN
IF(INCX==1)GO TO 20
!
!        CODE FOR INCREMENT NOT EQUAL TO 1
!
IX = 1
SMAX = (SX(1))
IX = IX + INCX
DO 10 I = 2,N
   IF((SX(IX))<=SMAX) GO TO 5
   ismaxi = I
   SMAX = (SX(IX))
5  IX = IX + INCX
10 CONTINUE
RETURN
!
!        CODE FOR INCREMENT EQUAL TO 1
!
20 SMAX = (SX(1))
DO 30 I = 2,N
   IF((SX(I))<=SMAX) GO TO 30
   ismaxi = I
   SMAX = (SX(I))
30 CONTINUE
RETURN
   end function ismaxi
