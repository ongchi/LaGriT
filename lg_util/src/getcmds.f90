!dk,getcmds
subroutine getcmds(n,imsgout,xmsgout,cmsgout,msgtype,nwds, &
                        imsgprmp)
!
!#######################################################################
!
!      PURPOSE -
!
!         THIS ROUTINE GETS AND PROCESS COMMANDS EITHER IN AN
!            INTERACTIVE MODE OR FROM A COMMAND FILE.
!
!      INPUT ARGUMENTS -
!
!         NONE
!
!
!      OUTPUT ARGUMENTS -
!
!         NONE
!
!
!      CHANGE HISTORY -
!
!        $Log: getcmds.f,v $
!        Revision 2.00  2007/11/03 00:49:10  spchu
!        Import to CVS
!
!PVCS    
!PVCS       Rev 1.18   01 Jun 2001 16:29:00   dcg
!PVCS    get rid of upper case
!PVCS    make implicit none
!PVCS
!PVCS       Rev 1.17   Tue Apr 13 10:49:20 1999   dcg
!PVCS    changes for new parser
!PVCS
!PVCS       Rev 1.16   Sat Jan 10 07:08:24 1998   het
!PVCS    Add the getcmds_msg command to retrive the next message
!PVCS       from whatever source is available.
!PVCS
!PVCS       Rev 1.15   Thu Sep 18 13:29:04 1997   het
!PVCS    Correct an error with the length of cmsgout(1)
!PVCS
!PVCS       Rev 1.14   Mon Sep 15 13:18:42 1997   het
!PVCS    Add the title and comment card options.
!PVCS
!PVCS       Rev 1.12   Wed Mar 06 16:28:48 1996   dcg
!PVCS    include string length in calls to parse_string
!PVCS
!PVCS       Rev 1.11   12/08/95 14:05:32   dcg
!PVCS    fix blank before continuation
!PVCS
!PVCS       Rev 1.10   11/27/95 11:10:34   het
!PVCS    Make UNICOS changes
!PVCS
!PVCS       Rev 1.9   05/24/95 15:46:10   het
!PVCS    Change the write format for file names
!PVCS
!PVCS       Rev 1.8   03/21/95 12:31:14   dcg
!PVCS
!PVCS       Rev 1.7   02/23/95 13:43:02   ejl
!PVCS    Fisxed problem with blank line input from TTy.
!PVCS    Also fixed default option COMMAND/////input/.
!PVCS
!PVCS       Rev 1.6   02/23/95 13:12:40   ejl
!PVCS    Fixed problem with Blank Line input.
!PVCS
!PVCS       Rev 1.5   02/07/95 13:06:38   het
!PVCS    Correct an error with ibuff (-1)
!PVCS
!PVCS       Rev 1.4   01/23/95 17:30:46   dcg
!PVCS
!PVCS
!PVCS       Rev 1.3   01/04/95 22:00:06   llt
!PVCS    unicos changes (made by het)
!PVCS
!PVCS       Rev 1.2   01/02/95 10:54:32   het
!PVCS    Change the file name references to 32 characters instead
!PVCS    of 4096 characters (this gives the SGI trouble).
!PVCS
!PVCS
!PVCS       Rev 1.1   12/21/94 11:14:36   het
!PVCS    Corrected an error with adding char(0) as the terminating character.
!PVCS
!PVCS
!PVCS       Rev 1.0   11/10/94 12:41:48   pvcs
!PVCS    Original version.
!
!#######################################################################
implicit none
!
!     The I/O library does not have entry points for handling
!     type INTEGER*8 I/O control list specifiers
integer*4 junitcmd
!
!     THE FOLLOWING ARRAYS ARE FOR JACK PETERSON'S PARSER.
!
integer nd,nsd,nwrds,nchrwrd,lmsgt,n,imsgout(*), &
       msgtype(*),nwds,lenend,nwds1,nwds2,ib2,jname, &
       ierrass,lenstart,ib,li,loc,locp,locm,ist,iet1, &
       ibt1,ierrlog,nummsg,lenparse,ipos,iflag,ie,i, &
       icount,iflag2,ist1,ist2,length2,length1,idum, &
       length,icode,icharlnb,ie1,ib1,icont, &
      icharlnf,lenbuff,i1,lmsg,nwdsold, &
      iqchrn,itthr,ipause,istack,ictcmd, &
      icommand,jcommand,lcommand,len1,iecho,ierr,isvpause, &
      nchrsav,imsgold
 
PARAMETER ( nd=4 , nsd=1 )
common /getcmdsd/ id(nd)
character*1 id
!
character*1 kchar
!
!     THESE ARRAYS CONTAIN THE COMPLETE MESSAGE AS READ FROM THE
!     TERMINAL
!
PARAMETER ( nwrds = 128 , nchrwrd = 8 )
PARAMETER ( lmsgt=nwrds*nchrwrd)
!
!     THESE ARRAYS CONTAIN THE LAST MESSAGE PROCESSED SO THAT IT CAN BE
!     REPEATED.
!
COMMON /GETCMDS2/ lmsg
COMMON /GETCMDSA/ msgb, msgc
character*4096 msgb, msgc
COMMON /GETCMDS3/ nwdsold, imsgold(nwrds)
COMMON /GETCMDSB/ msgold
character*4096 msgold
!
!     THESE ARRAYS WILL CONTAIN THE INDIVIDUAL COMMANDS THAT ARE PULLED
!     OFF STACKS.
!
real*8 xmsgout(n)
character*32 cmsgout(n), ccommand
!
!     THIS COMMON PROVIDES SPACE FOR A CFTLIB MESSAGE BUFFER.
!     NOTE: THIS BUFFER IS 64 WORDS LONG FOLLOWED BY 1 WORD THAT
!           INDICATES HOW MANY CHARACTERS ARE IN THE BUFFER.  IF THE
!           FIRST WORD EQUALS -1, THEN NO MESSAGE IS PRESENT.
!
 
COMMON /QMBUFFC0/ iqchrn
COMMON /QMBUFFC1/ ibuff
character*4096 ibuff
!
COMMON /MSGSAVE0/ nchrsav
COMMON /MSGSAVE1/ msgsav
character*4096 msgsav
!
COMMON /GETCMDS4/ itthr, ipause, istack
!
COMMON /GETCMDSE/ kcommand(10)
character*32 kcommand, iname
COMMON /GETCMDS5/ ictcmd, icommand, jcommand, lcommand
!
!     PROVIDE SPACE (IN LOW-CORE [<2,000,000]) FOR AN I/O BUFFER.
!
COMMON /GETCOM1/ isvpause
!
!#######################################################################
!
CHARACTER*132 interfil
CHARACTER*80 imessage
CHARACTER imsgprmp*(*)
character*80 iline
character*32 ifile
!
!#######################################################################
!
!CHT DECLARE THE FUNCTION ALLBLK (ALL BLANK) TO BE LOGICAL
!
!
!#######################################################################
!
!
!#######################################################################
!
!
!     ******************************************************************
!
!     INITIALIZE VARIABLES IN COMDECK getcmdsc.
!
data iecho / 1 /
!
!#######################################################################
!
!cht
!cht
!cht
!cht set the prompt message
!cht
imessage='nomessage'
if( imsgprmp /= ' ' ) imessage = imsgprmp
!cht
imessage='nomessage'
if( imsgprmp /= ' ' ) imessage = imsgprmp
!cht
!cht
if(itthr==0) then
   itthr=1
   ictcmd=0
   ipause=0
   istack=0
   icommand=0
   jcommand=0
   lcommand=0
   id(1)=','
   id(2)='/'
   id(3)='='
   id(4)=':'
end if
!cht
goto 101
goto 9998
101 continue
!cht
!cht blank out the message array
!cht
nwds=0
do 200 i1=1,n
   msgtype(i1)=0
   imsgout(i1)=0
   xmsgout(i1)=0.0d+00
   cmsgout(i1)=' '
200 continue
msgc=' '
!cht
if(istack==-1) then
   istack=0
   goto 100
else if(istack==0) then
   lenbuff=icharlnf(ibuff)
   if(jcommand==0.or.ipause/=0) then
      if(imessage=='nomessage') then
         imessage='enter an interactive command'
      end if
!
!
!    GET COMMANDS IN AN INTERACTIVE MODE.
!
      if(ibuff(1:lenbuff)=='-1') then
         if(imessage=='*empty*') goto 100
            write(interfil,9000) imessage
            call writloga('default',1,interfil,0,ierr)
         icont=0
220      continue
!*****         call rcvtty
         read(*,'(a80)',end=100) ibuff
221      continue
         icont=index(ibuff,'&')
         if(icont>0) then
            ibuff(icont:icont)=' '
            do while(ibuff((icont-2):(icont-2))==' ')
              icont=icont-1
            end do
            read(*,'(a80)') iline
            ib1=1
            ie1=icharlnb(iline)
            do while(iline(ib1:ib1)==' ')
               ib1=ib1+1
            end do
            ibuff(icont:icont+ie1-ib1)=iline(ib1:ie1)
            if(icont>0) goto 221
         end if
         iqchrn=icharlnb(ibuff)
         if(ibuff(1:iqchrn)=='-1') goto 220
      end if
      icode=0
      lmsg=iqchrn
      msgb = ' '
      msgb(1:iqchrn)=ibuff(1:iqchrn)
      ibuff='-1'
      if(icode==2) goto 101
   else if(jcommand>0 .and. ibuff(1:lenbuff)/='-1') then
      msgb = ' '
      msgb=ibuff(1:iqchrn)
      lmsg = iqchrn
      ibuff='-1'
   else
      ipause=0
      jcommand=2
      ifile=kcommand(lcommand)
      length=icharlnf(ifile)
      inquire(file=ifile(1:length),number=junitcmd)
!              *** GET THE LOGICAL UNIT NUMBER ASSOCAITED WITH THIS FILE
      icode=0
      msgb=' '
      read(junitcmd,'(a80)',end=202) msgb(1:80)
222   continue
      icont=index(msgb,'&')
      if(icont>0) then
         msgb(icont:icont)=' '
         do while(msgb((icont-2):(icont-2))==' ')
            icont=icont-1
         end do
         read(junitcmd,'(a80)',end=202) iline
         ib1=1
         ie1=icharlnb(iline)
         do while(iline(ib1:ib1)==' ')
            ib1=ib1+1
         end do
         msgb(icont:icont+ie1-ib1)=iline(ib1:ie1)
         goto 222
      end if
      goto 201
202   continue
      icode=2
201   continue
      iqchrn=icharlnb(msgb)
      do idum=1,iqchrn
         if(msgb(iqchrn:iqchrn)==' ') then
            iqchrn=iqchrn-1
            goto 203
         end if
      end do
203   continue
      lmsg=iqchrn
      if(msgb(1:7)=='endfile') icode=2
      if(icode==2) then
         close(junitcmd)
         write(interfil,9010) lcommand, &
              kcommand(lcommand)(1:icharlnf(kcommand(lcommand)))
         call writloga('default',1,interfil,0,ierr)
         lcommand=lcommand-1
         if(lcommand==0) then
            icommand=0
            jcommand=0
            ipause=isvpause
            if(ipause==0) then
               goto 100
            else
               goto 101
            end if
         else
            write(interfil,9011) lcommand, &
                    kcommand(lcommand)(1:icharlnf(kcommand(lcommand)))
            call writloga('default',0,interfil,1,ierr)
            goto 101
         end if
      end if
   end if
end if
lenbuff=icharlnf(ibuff)
 if(ibuff(1:lenbuff)/='-1') then
    length1=iqchrn
    length2=length1+4
    msgb(length2:(length2+lmsg))=msgb(1:lmsg)
    msgb(1:length1)=ibuff(1:length1)
    msgb((length1+1):(length1+1+3))=' ; '
    lmsg = lmsg + length1 + 3
    ibuff='-1'
 end if
 if(lmsg<=0) goto 101
 do 120 i1=1,nd
    ist2=1
110 continue
    ist1=index(msgb(ist2:lmsg),id(i1))
    if(ist1==0) then
       ist1=lmsg+1
    else
       ist1=ist1 + (ist2-1)
    end if
    if(ist1>=lmsg) goto 115
    ist2=index(msgb(ist1+1:lmsg),id(i1))
    if(ist2==0) then
       ist2=lmsg+1
    else
       ist2=ist2 + ist1
    end if
    if(ist2>lmsg) goto 115
    iflag2=0
    if(ist1+1==ist2) iflag2=1
    icount=0
    do i=ist1+1,ist2-1
       if(msgb(i:i)==' ') icount=icount+1
    end do
    if(icount==(ist2-1-ist1)) iflag2=1
    if(iflag2==1) then
       msgc=' -def-'
       ie=lmsg-ist1+6
       msgc(7:ie)=msgb((ist1+1):lmsg)
       lmsg=lmsg+6-1
       msgb(ist1:lmsg)=msgc(1:ie)
    end if
    goto 110
115 continue
    iflag=0
    do 140 i=1,lmsg
       if(msgb(i:i)=="'".or.msgb(i:i)=="'") iflag=iflag+1
       if(mod(iflag,2)==0.and.msgb(i:i)==id(i1)) msgb(i:i)=' '
140 continue
120 continue
ipos=index(msgb,';')
if(ipos==0) ipos=lmsg+1
if(ipos<lmsg) then
   istack=1
   length=ipos-1
   msgc=' '
   msgc(1:length)=msgb(1:length) // char(0)
lenparse=length
   length=lmsg-ipos
   msgold=' '
   msgold(1:length)=msgb((ipos+1):lmsg) // ' '
   msgb=' '
   msgb(1:length)=msgold(1:length)
   lmsg=lmsg-ipos
else
   if(jcommand==0) then
      if(ipause==0) then
         istack=-1
      else
         istack=0
      end if
   else
      istack=0
   end if
   msgc(1:lmsg)=msgb(1:lmsg) // char(0)
  lenparse=lmsg
end if
ictcmd=ictcmd+1
nummsg=int((ipos-1)/nchrwrd)+1
if(iecho==1) then
   write(interfil,9040) ictcmd,msgc(1:8*min(9,nummsg))
   call writloga('default',0,interfil,0,ierr)
   if(nummsg>9) then
      do 102 i=10,nummsg,9
         write(interfil,9041) msgc(8*(i-1)+1:8*min(i+8,nummsg))
         call writloga('default',0,interfil,0,ierr)
102   continue
   end if
end if
interfil=' '
call writset('stat','log','on',ierrlog)
if(nummsg>9) then
   write(interfil,9042) msgc(1:8*min(9,nummsg))
   call writloga('log',0,interfil,0,ierr)
   do 103 i=10,nummsg,9
      if(nummsg>(i+8)) then
         write(interfil,9044) msgc(8*(i-1):8*min(i+8,nummsg))
         call writloga('log',0,interfil,0,ierr)
      else
         write(interfil,9045) msgc(8*(i-1):8*min(i+8,nummsg))
         call writloga('log',0,interfil,0,ierr)
      end if
103 continue
else
   write(interfil,'(a6)') msgc(1:6)
   if(interfil(1:6)=='infile') then
      write(interfil,9046) msgc(1:8*min(9,nummsg))
      call writloga('log',0,interfil,0,ierr)
   else
      write(interfil,9043) msgc(1:8*min(9,nummsg))
      call writloga('log',0,interfil,0,ierr)
   end if
end if
call writfls('log',ierrlog)
call writset('stat','log','off',ierrlog)
call writfls('bat',ierrlog)
!
!
!     CHANGE LOWER CASE E IN NUMBERS TO UPPER CASE; PARSER
!     WON'T ACCEPT LOWER CASE E'S.
!
!
ibt1=1
iet1=ipos-1
ist=ibt1
150 continue
!
!
!     LOOK FOR AN 'e'
!
!
locm=index(msgc(ist:iet1),'e-')
locp=index(msgc(ist:iet1),'e+')
if(locm>0.or.locp>0) then
   loc=max(locm,locp)
   loc = loc + (ist-1)
!
!
!
!     MARCH BACKWARDS UNTIL WE GET TO A COMMAND DELIMITER
!     OR TO SOMETHING THAT SAYS THIS ISN'T A NUMBER; I.E.,
!     NOT A DIGIT, PLUS, MINUS, OR PERIOD.
!
!
   do 151 li=loc-1,ist,-1
      kchar=msgc(li:li)
!
!
!           IF IT'S A COMMAND DELIMITER:
!
!
      if (kchar==' ') goto 152
!
!
!           IF IT ISN'T A DIGIT OR A PERIOD OR A PLUS OR MINUS
!
!
      if((kchar>'9').or.(kchar<'+').or. &
              (kchar==',').or.(kchar=='/')) goto 153
151 continue
!
!
!     WE GOT TO A COMMAND DELIMITER WITHOUT FINDING
!     ANYTHING BUT DIGITS, +, -, OR ..  SO WE WILL
!     REPLACE THE 'e' WITN AN 'E'.
!
!
!     IF 'e' WAS FIRST CHAR, SKIP
!
152 if (li==loc-1) goto 153
!
!
   ib=loc-1
   ie=ib+1
   msgc(ie:ie)='E'
!
!
!     UPDATE STRING START POINTER; IF NOT AT END, GO BACK
!     AND SEARCH FOR MORE.
!
!
   ist=loc+1
   loc=index(msgc(ist:iet1),' ')
   if(loc>0) then
      ist=loc + 1 + (ist-1)
      if (ist<iet1) goto 150
   end if
153 continue
end if
!
!      SAVE THE UNPARSED MESSAGE JUST IN CASE WE NEED TO GET
!      AT IT FOR RE-PARSING.  THIS MIGHT OCCUR IF WE DON'T
!      LIKE THE FACT THAT THIS ROUTINE REMOVE SEVERAL SPECIAL
!      CHARACTERS AS DELIMITERS.
!
nchrsav=iet1-ibt1+1
length=nchrsav-ibt1+1
msgsav(1:length)=msgc(ibt1:nchrsav)
!
!
!*****call gparse(pcb,imsgout,msgtype,nwds,ibt2,iet2,msgc,ibt1,iet1)
call parse_string(lenparse,msgc,imsgout,msgtype,xmsgout,cmsgout, &
                   nwds)
!
ccommand=cmsgout(1)
!cht
!cht
!cht repeat the last message
!cht
!cht
!cht process a 'run' command (this is different that 'go' or
!cht    'continue' since this command interactes with the driving
!cht    code and and the command processor).
!cht
if(ccommand(1:3)=='run') then
   ipause=0
   if(jcommand/=0) then
      ifile=kcommand(lcommand)
      length=icharlnf(ifile)
      inquire(file=ifile(1:length),number=junitcmd)
!              *** GET THE LOGICAL UNIT NUMBER ASSOCAITED WITH THIS FILE
      close(junitcmd)
      write(interfil,9010) lcommand,kcommand(lcommand)
      call writloga('default',2,interfil,0,ierr)
      lcommand=lcommand-1
      if(lcommand==0) then
         icommand=0
         jcommand=0
         ipause=isvpause
      end if
   end if
end if
!cht
!cht
!cht process a 'go' command
!cht
if(ccommand(1:8)=='continue') then
   ipause=0
   goto 101
end if
!cht
!cht
!cht process a message enclosed in quotes
!cht
if((ccommand(1:5)=='title' .or. ccommand(1:7)=='comment').and. &
        lenparse>7) then
   lenstart=0
   lenend=0
   do i=1,lenparse
      if(msgb(i:i)=="'") then
         if(lenstart==0) then
            lenstart=i+1
         else
            lenend=i-1
         end if
      end if
   end do
   len1=len(cmsgout(1))
   nwds1=1+(lenend-lenstart)/len1
   nwds2=lenend-lenstart-len1*(nwds1-1)
   ib1=lenstart
   ib2=min(ib1+len1-1,lenend)
   nwds=nwds1
   do i=2,nwds1+1
      msgtype(i)=3
      cmsgout(i)=msgb(ib1:ib2)
      ib1=ib2+1
      ib2=min(ib2+len1,lenend)
   end do
   cmsgout(nwds1+1)(ib1:len1*(nwds1+1)) = &
           '                                '
   goto 9998
9998 continue
end if
!cht
!cht
!cht see if this is the 'command file' command
!cht
if(ccommand(1:6)=='infile'.or.ccommand(1:5)=='input') then
   if(nwds<2) then
      write(interfil,9030)
      call writloga('default',1,interfil,0,ierr)
      goto 101
   end if
   iname=cmsgout(2)
!********call fexist(iname,ierr)
   ierr=1
   if(ierr==0) then
      write(interfil,9050) iname
      call writloga('default',1,interfil,1,ierr)
   else
      jname=-1
      call hassign(jname,iname,ierrass)
      lcommand=lcommand+1
      kcommand(lcommand)=iname
      if(lcommand==1) then
         icommand=1
         jcommand=1
         isvpause=ipause
         ipause=0
      end if
      write(interfil,9020) lcommand, &
              kcommand(lcommand)(1:icharlnf(kcommand(lcommand)))
      call writloga('default',1,interfil,0,ierr)
   end if
   goto 101
end if
!cht
if(ccommand(1:3)=='end') goto 9998
!cht
if(ccommand(1:5)=='pause') then
   ipause=1
   if(jcommand==0) then
      nwds=0
      goto 100
   else
      goto 101
   end if
end if
!cht
goto 100
100 continue
!cht
if(istack==-1) istack=0
!cht
!cht
!cht zero out the prompt message array
!cht
if(istack==0.and.ipause==0.and.jcommand==0) then
   imsgprmp='*nomore*'
else
   imsgprmp='*more*'
end if
!cht
!cht
goto 9999
!
! Format statements
!
9000 format(' ',a80)
9010 format(' ','Closing command file:  ',i1,' - ',a)
9011 format(' ','Continue reading from:  ',i1,' - ',a)
9020 format(' ','Assign command file:  ',i5,' - ',a)
9030 format(' ','The command file name must be entered:')
9040 format(' ',i5,2x,a)
9041 format(' ',7x,a)
9042 format(a," &")
9043 format(a)
9044 format(3x,a," &")
9045 format(3x,a)
9046 format('* ',a)
9050 format(' ','The file:  ',a8, &
              ' does not exist as a local file')
!
9999 continue
return
   end subroutine getcmds
