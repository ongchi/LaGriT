!     File contains parse_string and parse_string2

subroutine parse_string(imsgout,msgtype, &
           xmsgout,cmsgout,nwds)
! 
!      Uses syntax defined for LAGRIT command lines
!      break up command, from commands_lg.h, into nwds tokens
!      put integers in imsgout, reals in xmsgout, strings in
!      cmsgout, msgtype is token type
!
! ######################################################################
!
!        $Log: parse_string.f,v $
!        Revision 2.00  2007/11/03 00:49:12  spchu
!        Import to CVS
!
!PVCS
!PVCS       Rev 1.21   02 Oct 2007 12:40:28   spchu
!PVCS    original version
!
! ######################################################################
!
implicit none

include 'commands_lg.h'

integer lenparse,nwds
integer imsgout(*),msgtype(*)
real*8 xmsgout(*)
character*32 cmsgout(*)

integer i,j,istrt,length,inum,k,icharlnb
logical isintgr,isreal
real*8 xnum

lenparse=icharlnb(command)
!
nwds=0
inum=0
xnum=0.0d0
!  skip leading blanks and tabs
   istrt=1
   j=1
   do while (command(j:j)==' ' .or. &
           command(j:j)==achar(9))
      j=j+1
   end do
   istrt=j

!  look for delimiter break off a token
10 i=0
   if(istrt>lenparse) go to 100
   nwds=nwds+1
   cmsgout(nwds)=' '
   j=istrt
   do while (command(j:j)/=' '.and.command(j:j)/=','.and. &
          command(j:j)/='='.and. command(j:j)/=achar(9) .and. &
          command(j:j)/=';'.and. command(j:j)/='/' .and. &
          command(j:j)/=char(0) .and. j<=lenparse)
     i=i+1
     cmsgout(nwds)(i:i)=command(j:j)
     j=j+1
     if(j>lenparse) go to 20
   end do
!  figure out what kind of token
20 length=i
   msgtype(nwds)=3
   imsgout(nwds)=0
   xmsgout(nwds)=0.
!  if length is zero fill in the default token
   if(length==0.or.cmsgout(nwds)(1:length)==' ') then
      cmsgout(nwds)='-def-'
      go to 25
   end if
   inum=0
   xnum=0.0d0
   call isinteger_lg(cmsgout(nwds),length,inum,isintgr)
   if(isintgr) then
      imsgout(nwds)=inum
      xmsgout(nwds)=inum
      msgtype(nwds)=1
   else
      call isrl_lg(cmsgout(nwds),length,xnum,isreal)
      if(isreal) then
          xmsgout(nwds)=xnum
          msgtype(nwds)=2
      end if
   end if
!  look for next token
25 istrt=j+1
   j=istrt
   if(j>lenparse) go to 100
   do while (command(j:j)==' ' .or. &
                  command(j:j)==achar(9) )
      j=j+1
      if(j>lenparse) go to 100
   end do
   istrt=j
   go to 10
100 continue
return
end subroutine parse_string

!
!     evaluate word, return inum and isintgr 
subroutine isinteger_lg(word,len,inum,isintgr)

implicit none

integer len,inum
character*(*) word
logical isintgr

integer i,j,iflag,m,itop,ibot

inum=0
isintgr=.true.

!  integer is +- numbers
iflag=1
j=1
if(word(1:1)=='-') then
   iflag=-1
   j=2
else if(word(1:1)=='+') then
   j=2
end if
do i=j,len
    m=ichar(word(i:i))
    itop=ichar('9')
    ibot=ichar('0')
    if(m>=ibot.and.m<=itop) then
      inum=inum*10 + m-ibot
    else
      isintgr=.false.
      go to 9999
    end if
 end do
 isintgr=.true.
 inum=iflag*inum

9999 continue
 return
 end subroutine isinteger_lg

!     evaluate tokens with scientific notation     
!     return xnum and isreal
subroutine isrl_lg(word,len,xnum,isreal)

implicit none

integer len
character*(*) word
real*8 xnum
logical isreal

integer inum,j,iflag,ii,m,ibot,itop
real*8 power,ten

isreal=.false.
ten=10.0d0
xnum=0.0d0
iflag=1
j=1
if(word(1:1)=='-') then
  iflag=-1
  j=j+1
else if (word(1:1)=='+') then
  j=j+1
end if
!  look for base
inum=0
ii=j
itop=ichar('9')
ibot=ichar('0')
m=ichar(word(ii:ii))
if ((m>itop.or.m<ibot).and.word(ii:ii)/='.') then
   isreal=.false.
   return
end if
do while &
       (m>=ibot.and.m<=itop)
   inum=inum*10+m-ichar('0')
   ii=ii+1
   m=ichar(word(ii:ii))
end do
if(word(ii:ii)=='.') then
!  get fraction part of base
   ii=ii+1
   if(ii>len) then
      isreal=.true.
      xnum=inum*iflag
      go to 9999
   end if
   xnum=inum
   power=.1d0
   m=ichar(word(ii:ii))
   do while &
           (m>=ibot.and.m<=itop)
         xnum=xnum+(m-ibot)*power
         power=power*.1d0
         ii=ii+1
         m=ichar(word(ii:ii))
   end do
   xnum=xnum*iflag
   if(ii>len) then
      isreal=.true.
      go to 9999
   end if
end if
if(word(ii:ii)=='e'.or.word(ii:ii)=='E'.or. &
        word(ii:ii)=='+'.or.word(ii:ii)=='-'.or. &
        word(ii:ii)=='d'.or.word(ii:ii)=='D') then
if(ii==len) go to 9999
   if(xnum==0.0d0.and.inum==0.and.ii==1) go to 9999
   if(xnum==0d0) xnum=inum*iflag
   iflag=1
   if(word(ii:ii)=='-') iflag=-1
!  get exponent
   ii=ii+1
   inum=0
   if(word(ii:ii)=='+') ii=ii+1
   if(word(ii:ii)=='-') then
      ii=ii+1
      iflag=-1
   end if
   m=ichar(word(ii:ii))
   do while &
          (m>=ibot.and.m<=itop.and. &
          ii<=len)
       inum=inum*10+m-ichar('0')
       ii=ii+1
       m=ichar(word(ii:ii))
   end do
   if(ii<=len) go to 9999
   if(iflag==1) then
      xnum=xnum*ten**inum
    else
      xnum=xnum/ten**inum
   end if
   isreal=.true.
end if
9999 continue
   return
   end subroutine isrl_lg

!     parse_string2
!     this version takes a single line msgc and parses into 
!     nwds number of tokens seperated by white space 
!     with the appropriate type assignments
!     put integers in imsgout, reals in xmsgout, strings in
!     cmsgout, msgtype is token type

subroutine parse_string2(lenparse,msgc,imsgout,msgtype, &
           xmsgout,cmsgout,nwds)

implicit none

!     arguments
integer lenparse,nwds
character*4096 msgc
integer imsgout(*),msgtype(*)
real*8 xmsgout(*)
character*32 cmsgout(*)

integer i,j,istrt,length,inum
logical isintgr,isreal
real*8 xnum
!
nwds=0
length=0
inum=0
xnum=0.0d0

!  skip leading blanks
istrt=1
j=1
do while (msgc(j:j)==' ' .or. msgc(j:j)==achar(9))
   j=j+1
end do

!  look for delimiter break off a token
istrt=j
10 i=0
if(istrt>lenparse) go to 100
nwds=nwds+1
cmsgout(nwds)=' '
j=istrt
do while ( msgc(j:j)/=' '.and.msgc(j:j)/=','.and. &
         msgc(j:j)/=';'.and.msgc(j:j)/='%'.and. &
         msgc(j:j)/=achar(9) .and. &
         msgc(j:j)/=char(0) .and. j<=lenparse)
  i=i+1
  if (i==32) then
     nwds=nwds+1
     cmsgout(nwds)=' '
     i=1
  end if
  cmsgout(nwds)(i:i)=msgc(j:j)
  j=j+1
  if(j>lenparse) go to 20
end do
!  figure out what kind of token
20 length=i
msgtype(nwds)=3
imsgout(nwds)=0
xmsgout(nwds)=0.
inum=0
xnum=0.0d0
call isinteger_lg(cmsgout(nwds),length,inum,isintgr)
if(isintgr) then
   imsgout(nwds)=inum
   xmsgout(nwds)=inum
   msgtype(nwds)=1
else
   call isrl_lg(cmsgout(nwds),length,xnum,isreal)
   if(isreal) then
       xmsgout(nwds)=xnum
       msgtype(nwds)=2
   end if
end if
!  look for next token
istrt=j+1
j=istrt
if(j>lenparse) go to 100
do while (msgc(j:j)==' ' .or. msgc(j:j)==achar(9))
   j=j+1
   if(j>lenparse) go to 100
end do
istrt=j

go to 10

100 continue
return
   end subroutine parse_string2

!     End file
