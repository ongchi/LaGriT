SUBROUTINE isort(X,Y,N,KFLAG)
!
! #####################################################################
!
!     PURPOSE -
!
!        None
!
!     INPUT ARGUMENTS -
!
!        None
!
!     OUTPUT ARGUMENTS -
!
!        None
!
!     CHANGE HISTORY -
!
!        $Log: isort.f,v $
!        Revision 2.00  2007/11/03 00:49:11  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.4   01 Jun 2001 16:29:42   dcg
!PVCS    make name lower case
!PVCS
!PVCS       Rev 1.3   08/03/95 13:53:22   dcg
!PVCS    replace print * with writloga calls
!PVCS
!PVCS       Rev 1.2   01/20/95 12:21:52   dcg
!PVCS     declare r to be real*4 for type compatibility
!PVCS
!PVCS       Rev 1.1   01/04/95 21:55:42   llt
!PVCS    unicos changes (made by het)
!PVCS
!PVCS       Rev 1.0   11/10/94 12:42:36   pvcs
!PVCS    Original version.
!
! ######################################################################
!
implicit real*8 (a-h,o-z)
!***BEGIN PROLOGUE  ISORT
!***DATE WRITTEN   761118   (YYMMDD)
!***REVISION DATE  861211   (YYMMDD)
!***CATEGORY NO.  N6A2A
!***KEYWORDS  LIBRARY=SLATEC,TYPE=INTEGER(SSORT-S DSORT-D ISORT-I),
!             QUICKSORT,SINGLETON QUICKSORT,SORT,SORTING
!***AUTHOR  JONES, R. E., (SNLA)
!           KAHANER, D. K., (NBS)
!           WISNIEWSKI, J. A., (SNLA)
!***PURPOSE  ISORT sorts integer array X and optionally makes the same
!            interchanges in integer array Y.  The array X may be
!            sorted in increasing order or decreasing order.  A
!            slightly modified QUICKSORT algorithm is used.
!***DESCRIPTION
!
!     Written by Rondall E Jones
!     Modified by John A. Wisniewski to use the Singleton QUICKSORT
!     algorithm. Date 18 November 1976.
!
!     Further modified by David K. Kahaner
!     NATIONAL BUREAU OF STANDARDS
!     August, 1981
!
!     Abstract
!         ISORT sorts integer array X and optionally makes the same
!         interchanges in integer array Y.  The array X may be sorted in
!         INCREASING order or DECREASING order.  A slightly modified
!         QUICKSORT algorithm is used.
!
!     Reference
!         Singleton,R.C., Algorithm 347, An Efficient Algorithm For
!         Sorting With Minimal Storage, CACM,12(3),1969,185-7.
!
!     Description of Parameters
!         X - integer array of values to be sorted
!         Y - integer array to be (optionally) carried along
!         N - number of values in integer array X to be sorted
!     KFLAG - control parameter
!           = 2 means sort X in INCREASING order and carry Y along.
!           = 1 means sort X in INCREASING order (ignoring Y)
!           =-1 means sort X in DECREASING order (ignoring Y)
!           =-2 means sort X in DECREASING order and carry Y along.
!***REFERENCES  SINGLETON, R. C., ALGORITHM 347, AN EFFICIENT
!                 ALGORITHM FOR SORTING WITH MINIMAL STORAGE, CACM,
!                 VOL. 12, NO. 3, 1969, PP. 185-187.
!***ROUTINES CALLED  XERROR
!***END PROLOGUE  ISORT
dimension IL(21),IU(21)
real*4 r
INTEGER X(N),Y(N),T,TT,TY,TTY
character*80 logmess
!***FIRST EXECUTABLE STATEMENT  ISORT
NN = N
IF (NN>=1) GO TO 10
   write(logmess,'(a)') &
           'SSORT- THE NUMBER OF VALUES TO BE SORTED IS NOT POSITIVE'
   call writloga('default',0,logmess,0,ierr)
RETURN
10 KK = IABS(KFLAG)
IF ((KK==1).OR.(KK==2)) GO TO 15
write(logmess,'(a)') &
       'SSORT- THE SORT CONTROL PARAMETER, K, WAS NOT 2, 1, -1, OR -2.'
call writloga('default',0,logmess,0,ierr)
RETURN
15 CONTINUE
!
! ALTER ARRAY X TO GET DECREASING ORDER IF NEEDED
!
IF (KFLAG>=1) GO TO 30
DO 20 I=1,NN
X(I) = -X(I)
20 CONTINUE
GO TO (100,200),KK
!
! SORT X ONLY
!
30 CONTINUE
100 CONTINUE
M=1
I=1
J=NN
R=.375
110 IF (I == J) GO TO 155
IF (R > .5898437) GO TO 120
R=R+3.90625E-2
GO TO 125
120 R=R-.21875
125 K=I
!                                  SELECT A CENTRAL ELEMENT OF THE
!                                  ARRAY AND SAVE IT IN LOCATION T
!*****IJ = I + IFIX (FLOAT (J-I) * sngl(R))
IJ = I + IFIX (FLOAT (J-I) * R)
T=X(IJ)
!                                  IF FIRST ELEMENT OF ARRAY IS GREATER
!                                  THAN T, INTERCHANGE WITH T
IF (X(I) <= T) GO TO 130
X(IJ)=X(I)
X(I)=T
T=X(IJ)
130 L=J
!                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
!                                  T, INTERCHANGE WITH T
IF (X(J) >= T) GO TO 140
X(IJ)=X(J)
X(J)=T
T=X(IJ)
!                                  IF FIRST ELEMENT OF ARRAY IS GREATER
!                                  THAN T, INTERCHANGE WITH T
IF (X(I) <= T) GO TO 140
X(IJ)=X(I)
X(I)=T
T=X(IJ)
GO TO 140
135 TT=X(L)
X(L)=X(K)
X(K)=TT
!                                  FIND AN ELEMENT IN THE SECOND HALF OF
!                                  THE ARRAY WHICH IS SMALLER THAN T
140 L=L-1
IF (X(L) > T) GO TO 140
!                                  FIND AN ELEMENT IN THE FIRST HALF OF
!                                  THE ARRAY WHICH IS GREATER THAN T
145 K=K+1
IF (X(K) < T) GO TO 145
!                                  INTERCHANGE THESE ELEMENTS
IF (K <= L) GO TO 135
!                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
!                                  THE ARRAY YET TO BE SORTED
IF (L-I <= J-K) GO TO 150
IL(M)=I
IU(M)=L
I=K
M=M+1
GO TO 160
150 IL(M)=K
IU(M)=J
J=L
M=M+1
GO TO 160
!                                  BEGIN AGAIN ON ANOTHER PORTION OF
!                                  THE UNSORTED ARRAY
155 CONTINUE
160 M=M-1
IF (M == 0) GO TO 300
I=IL(M)
J=IU(M)
IF (J-I >= 1) GO TO 125
IF (I == 1) GO TO 110
I=I-1
165 I=I+1
IF (I == J) GO TO 155
T=X(I+1)
IF (X(I) <= T) GO TO 165
K=I
170 X(K+1)=X(K)
K=K-1
IF (T < X(K)) GO TO 170
X(K+1)=T
GO TO 165
!
! SORT X AND CARRY Y ALONG
!
200 CONTINUE
M=1
I=1
J=NN
R=.375
210 IF (I == J) GO TO 255
IF (R > .5898437) GO TO 220
R=R+3.90625E-2
GO TO 225
220 R=R-.21875
225 K=I
!                                  SELECT A CENTRAL ELEMENT OF THE
!                                  ARRAY AND SAVE IT IN LOCATION T
!*****IJ = I + IFIX (FLOAT (J-I) *sngl(R))
IJ = I + IFIX (FLOAT (J-I) * R)
T=X(IJ)
TY= Y(IJ)
!                                  IF FIRST ELEMENT OF ARRAY IS GREATER
!                                  THAN T, INTERCHANGE WITH T
IF (X(I) <= T) GO TO 230
X(IJ)=X(I)
X(I)=T
T=X(IJ)
 Y(IJ)= Y(I)
 Y(I)=TY
TY= Y(IJ)
230 L=J
!                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
!                                  T, INTERCHANGE WITH T
IF (X(J) >= T) GO TO 240
X(IJ)=X(J)
X(J)=T
T=X(IJ)
 Y(IJ)= Y(J)
 Y(J)=TY
TY= Y(IJ)
!                                  IF FIRST ELEMENT OF ARRAY IS GREATER
!                                  THAN T, INTERCHANGE WITH T
IF (X(I) <= T) GO TO 240
X(IJ)=X(I)
X(I)=T
T=X(IJ)
 Y(IJ)= Y(I)
 Y(I)=TY
TY= Y(IJ)
GO TO 240
235 TT=X(L)
X(L)=X(K)
X(K)=TT
TTY= Y(L)
 Y(L)= Y(K)
 Y(K)=TTY
!                                  FIND AN ELEMENT IN THE SECOND HALF OF
!                                  THE ARRAY WHICH IS SMALLER THAN T
240 L=L-1
IF (X(L) > T) GO TO 240
!                                  FIND AN ELEMENT IN THE FIRST HALF OF
!                                  THE ARRAY WHICH IS GREATER THAN T
245 K=K+1
IF (X(K) < T) GO TO 245
!                                  INTERCHANGE THESE ELEMENTS
IF (K <= L) GO TO 235
!                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
!                                  THE ARRAY YET TO BE SORTED
IF (L-I <= J-K) GO TO 250
IL(M)=I
IU(M)=L
I=K
M=M+1
GO TO 260
250 IL(M)=K
IU(M)=J
J=L
M=M+1
GO TO 260
!                                  BEGIN AGAIN ON ANOTHER PORTION OF
!                                  THE UNSORTED ARRAY
255 CONTINUE
260 M=M-1
IF (M == 0) GO TO 300
I=IL(M)
J=IU(M)
IF (J-I >= 1) GO TO 225
IF (I == 1) GO TO 210
I=I-1
265 I=I+1
IF (I == J) GO TO 255
T=X(I+1)
TY= Y(I+1)
IF (X(I) <= T) GO TO 265
K=I
270 X(K+1)=X(K)
 Y(K+1)= Y(K)
K=K-1
IF (T < X(K)) GO TO 270
X(K+1)=T
 Y(K+1)=TY
GO TO 265
!
! CLEAN UP
!
300 IF (KFLAG>=1) RETURN
DO 310 I=1,NN
X(I) = -X(I)
310 CONTINUE
RETURN
   end subroutine isort
