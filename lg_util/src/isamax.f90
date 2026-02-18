!dk,isamax
integer function isamax(N,SX,INCX)
!
! ######################################################################
!
!        $Log: isamax.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS
!PVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
!PVCS    original version
!
! ######################################################################
!
implicit real*8 (a-h,o-z)
!  this function operates as double precision on short word machines
!***BEGIN PROLOGUE  isamax
!***DATE WRITTEN   791001   (YYMMDD)
!***REVISION DATE  861211   (YYMMDD)
!***CATEGORY NO.  D1A2
!***KEYWORDS  LIBRARY=SLATEC(BLAS),
!             TYPE=SINGLE PRECISION(isamax-S IDAMAX-D ICAMAX-C),
!             LINEAR ALGEBRA,MAXIMUM COMPONENT,VECTOR
!***AUTHOR  LAWSON, C. L., (JPL)
!           HANSON, R. J., (SNLA)
!           KINCAID, D. R., (U. OF TEXAS)
!           KROGH, F. T., (JPL)
!***PURPOSE  Find the smallest index of that component af a vector
!            having the maximum magnitude.
!***DESCRIPTION
!
!                B L A S  Subprogram
!    Description of Parameters
!
!     --Input--
!        N  number of elements in input vector(s)
!       SX  single precision vector with N elements
!     INCX  storage spacing between elements of SX
!
!     --Output--
!   isamax  smallest index (zero if N .LE. 0)
!
!     Find smallest index of maximum magnitude of single precision SX.
!     isamax =  first I, I = 1 to N, to minimize  ABS(SX(1-INCX+I*INCX)
!***REFERENCES  LAWSON C.L., HANSON R.J., KINCAID D.R., KROGH F.T.,
!                 *BASIC LINEAR ALGEBRA SUBPROGRAMS FOR FORTRAN USAGE*,
!                 ALGORITHM NO. 539, TRANSACTIONS ON MATHEMATICAL
!                 SOFTWARE, VOLUME 5, NUMBER 3, SEPTEMBER 1979, 308-323
!***ROUTINES CALLED  (NONE)
!***END PROLOGUE  isamax
!
real*8 SX(N),SMAX,XMAG
!***FIRST EXECUTABLE STATEMENT  isamax
isamax = 0
IF(N<=0) RETURN
isamax = 1
IF(N<=1)RETURN
IF(INCX==1)GOTO 20
!
!        CODE FOR INCREMENTS NOT EQUAL TO 1.
!
SMAX = ABS(SX(1))
NS = N*INCX
II = 1
    DO 10 I=1,NS,INCX
    XMAG = ABS(SX(I))
    IF(XMAG<=SMAX) GO TO 5
    isamax = II
    SMAX = XMAG
5   II = II + 1
10  CONTINUE
RETURN
!
!        CODE FOR INCREMENTS EQUAL TO 1.
!
20 SMAX = ABS(SX(1))
DO 30 I = 2,N
   XMAG = ABS(SX(I))
   IF(XMAG<=SMAX) GO TO 30
   isamax = I
   SMAX = XMAG
30 CONTINUE
RETURN
   end function isamax
