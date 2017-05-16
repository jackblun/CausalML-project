/*
Angrist-Chen: 11/21, 12/1, 12/3/2006 1/2/2007, 1/3
goal:DEFINE VARIABLES 
background:
 data constructed by getdata4.sas to extract variables from census 2000 
 selection rules- male residence in 51 states and pr 
1/7: check the following var def (to be comparable with ipums) 
 (1) SINGLE
 (2) VET75X90
 (3) VETOTH
 (4) WHITE
1/8: add ANYSERV
2/4: define sets of dummies for covariates, incl GQ, YOB, POB and Census regions
8/24/2009: add notes about the definitions of dimobile="physical dis" and disgoout="mobility dis"
*/
/*************/
/* libraries */
/*************/
options linesize=120;

libname incdat '/rdcprojects/br00487/data/extract';
libname savdat '/rdcprojects/br00487/data/workdat';

/********************************************************/
/* data step: variable definition and sample selection */
/********************************************************/

data savdat.vardef4;
set incdat.usmen01032007;

/********** VARIABLE DEFINITIONS ************************/
/* DEFINE AGE = age in census day */

AGE=agelong/1.;

/* DEFINE YOB FROM QDB */

YOB=substr(qdb,1,4)/1.;

/* DEFINE GQ=GROUP QUARTER */
/* rt  3  Housing unit person record */
/* 5 Group quarters person record */

if rt='5' then GQ=1;
 else if rt='3' then GQ=0;
 else GQ=.;

/* DEFIND POB FROM QPOBST */

POB=substr(qpobst, 1,3)/1.;

/* DEFINE SOR FROM PUID */

SOR=substr(puid, 1,2)/1.;

/* DEFINE RACE DUMMIES FROM SELF-CLAIMED VARIABLES */

if substr(qrace1,1,1)='1' then WHITE=1;
 else WHITE=0;

if substr(qrace1,1,1)='2' then BLACK=1;
 else BLACK=0;

if substr(imprace,2,1)='1' then WHITEI=1;
 else WHITEI=0;

if substr(imprace,2,1)='2' then BLACKI=1;
 else BLACKI=0;

/* DEFINE EDUCATIONAL LEVEL 
QATTEND School Enrollment (Attended School Since February 1, 2000) 0 Not in universe (less than 3 years old)
QATTEND School Enrollment (Attended School Since February 1, 2000) 1 No, has not attended since Feb. 1
QATTEND School Enrollment (Attended School Since February 1, 2000) 2 Yes, public school or college
QATTEND School Enrollment (Attended School Since February 1, 2000) 3 Yes, private school or college
*/

HGC=qhigh/1.;
IF HGC>=9 & HGC<=16 THEN HSGRAD=1;
 ELSE HSGRAD=0;
IF HGC>=10 & HGC<=16 THEN SOMCOL=1;
 ELSE SOMCOL=0;
IF HGC>=13 & HGC<=16 THEN COLGRA=1;
 ELSE COLGRA=0;
if qattend=2 or qattend=3 then nowschl=1;
 else nowschl=0;

/* DEFINE MARITAL STATUS */

msp=msp/1.;
IF msp=1 THEN MARRIEDP=1;
 ELSE MARRIEDP=0;
IF msp=2 THEN MARRIEDA=1;
 ELSE MARRIEDA=0;
IF msp=3 THEN WIDOWED=1;
 ELSE WIDOWED=0;
IF msp=4 THEN DIVORCED=1;
 ELSE DIVORCED=0;
IF msp=5 THEN SEPARATE=1;
 ELSE SEPARATE=0;
if msp=6 THEN SINGLE=1;
 ELSE SINGLE=0;

/* ANNUAL INCOME AND TOPCODING 
QINCWG: Wages/Salary Income in 1999 (Wages, salary, commissions, bonuses, or tips from all jobs) 
 TOP CODED = $999,999
QINCSE: Self-employment Income in 1999 (Self-employment income from own nonfarm businesses or 
 farm businesses, including proprietorships and partnerships)  
 TOP CODED = $999,999   LOWER CAP=-$9,998
QINCINT: Interest Income in 1999 (Interest, dividends, net rental income, royalty income, or 
 income from estates and trusts)
 TOP CODED = $99,999 LOWER CAP=-$9,998
QINCSS: Social Security Income in 1999 (Social Security or Railroad Retirement) 
 TOP CODED = $99,999
QINCSSI: Supplemental Security Income (SSI) in 1999 
 TOP CODED = $99,999
QINCPA: Public Assistance Income in 1999 (Any public assistance or welfare payments from 
 the state or local welfare office)
 TOP CODED = $99,999
QINCRET: Retirement Income in 1999 (Retirement, survivor, or disability pensions)
 TOP CODED = $999,999
QINCOTH: Other Income in 1999 (Any other sources of income received regularly such as 
 Veterans' (VA) payments, unemployment compensation, child support, or alimony) 
 TOP CODED = $99,999
QINCTOT: Total Income in 1999 (What was this person's total income in 1999?
 TOP CODED=$5,299,992  LOWER CAP=-$19,998
*/
/* POVERTY (POV)*/
/* HOURS PER WEEK (QWKLYRHR) */
/* WEEKS PER YEAR (QWKLYRWK) */
 
/* VETERAN STAUTUS (vps)
00 Not in universe (Persons under 18 years old/no active duty in the US Armed Forces)
01 August 1990 or later (including Persian Gulf War): Served in Vietnam era
02 August 1990 or later (including Persian Gulf War): No Vietnam era service: September 1980 or later only: Served under 2 years
03 August 1990 or later (including Persian Gulf War): No Vietnam era service: September 1980 or later only: Served 2 or more years
04 August 1990 or later (including Persian Gulf War): No Vietnam era service:  Served prior to September 1980
05 May 1975 to July 1990 only: September 1980 to July 1990 only: Served under 2 years
06 May 1975 to July 1990 only: September 1980 to July 1990 only: Served 2 or more years
07 May 1975 to July 1990 only: Other May 1975 to August 1980 service
08 Vietnam era, no Korean War, no WWII, no August 1990 or later
09 Vietnam era, Korean War, no WWII
10 Vietnam era, Korean War, and WWII
11 February 1955 to July 1964 only
12 Korean War, no Vietnam era, no WWII
13 Korean War and WWII, no Vietnam era
14 WWII, no Korean War, no Vietnam era
15 Other service only
QMIL1=1 if serve from April 1995 to 2000; 0 not
QMILAD Military Service: Ever Served on Active Duty in U.S. Armed Forces 0 Not in universe (age <17 years)
QMILAD Military Service: Ever Served on Active Duty in U.S. Armed Forces 1 Yes, now on active duty
QMILAD Military Service: Ever Served on Active Duty in U.S. Armed Forces 2 Yes, on active duty in the past, but not now
QMILAD Military Service: Ever Served on Active Duty in U.S. Armed Forces  3  No, training for Reserves or National Guard only
QMILAD Military Service: Ever Served on Active Duty in U.S. Armed Forces 4  No active duty service
*/
VPS=vps/1.;
IF VPS=1 OR (VPS>=8 & VPS<=10) THEN VIETNAM=1;
 ELSE VIETNAM=0;
IF VPS>=1 & VPS<=15 THEN ANYSERV=1;
 ELSE ANYSERV=0;

IF VPS>=2 & VPS<=4 THEN VET90X00=1;
 ELSE VET90X00=0;
IF VPS>=5 & VPS<=7 THEN VET75X90=1;
 ELSE VET75X90=0;
IF VPS=11 or VPS=15 THEN VETOTH=1;
 ELSE VETOTH=0; 
IF qmil1='1' then VET95X00=1;
 else VET95X00=0;
/* 
if vps>=2 & vps<=7 then postvet=1;
 else postvet=0; */
if vps>=1 & vps<=7 then postvet=1;
 else postvet=0; 
if qmilad =1 then nowmili=1;
 else nowmili=0;

/* DIABILITY
(DISABLE) 1 diable; 2 no
(QSENSE) 1 Sensory Difficulty: Have Long-lasting Vision or Hearing Impairment- 2 no
(QLMOB) 1 Physical Difficulty: Have Long-lasting Limited Mobility (e.g., walking, lifting)- 2 no
(QABMEN) 1 Mental Difficulty: Difficulty in Ability to Perform Mental Tasks (e.g., learning, remembering)- 2 no
(QABPHYS)1 Self-Care Difficulty: Difficulty in Ability to Dress, Bathe, Move About at Home- 2 no
(QABWORK) 1 Employment Difficulty: Difficulty in Ability to Work at a Job or Business- 2 no
(QABGO) 1 Outside Difficulty: Difficulty in Ability to Go Outside Home Alone (e.g., to shop)- 2 no
*/

IF disable='1' THEN DISABLE=1;
 ELSE DISABLE=0; 
IF qsense='1' THEN DISSENSE=1;
 ELSE DISSENSE=0; 
IF qlmob='1' THEN DISMOBILE=1; /* called "Physical disability" in the papers */ 
 ELSE DISMOBILE=0; 
IF qabmen='1' THEN DISMENTAL=1;
 ELSE DISMENTAL=0;
IF qabphys='1' THEN DISSELFCARE=1;
 ELSE DISSELFCARE=0;
IF qabwork='1' THEN DISWORK=1;
 ELSE DISWORK=0;
IF qabgo='1' THEN DISGOOUT=1; /* called "Mobility disability" in the papers */
 ELSE DISGOOUT=0;

/* LABOR FORCE STATUS 
(ESR) Employment Status Recode
0 Not in universe (less than 16 years old)
1 Employed, at work
2 Employed, with a job but not at work
3 Unemployed
4 Armed Forces, at work
5 Armed Forces, with a job but not at work
6 Not in labor force
(QLAYOFF) layoff from a job last week?
0 Not in universe (ESR=0,1, or 4)
1 Yes
2 No 
3 Not reported
(qlookwk) Looking for Work During Last 4 Weeks?
0 Not in universe (ESR=0,1, or 4)
1 Yes
2 No 
3 Not reported
*/
ESR=esr/1.;
IF ESR=6 THEN NLABORF=1;
 ELSE NLABORF=0;
IF ESR=1 or ESR=2 or ESR=4 or ESR=5 THEN EMPLOYED=1;
 ELSE  EMPLOYED=0;
IF ESR=3 THEN UNEMPLOYED=1;
 ELSE  UNEMPLOYED=0;

/* CLASS OF WORKER (QCOW)
0 Not in universe (less than 16 years old or did not work in the last 5 years)
1 Employee of PRIVATE FOR PROFIT
2 Employee of PRIVATE NOT-FOR-PROFIT
3 Employee of LOCAL GOVERNMENT
4 Employee of STATE GOVERNMENT
5 Employee of FEDERAL GOVERNMENT
6 SELF-EMPLOYED in NOT INCORPORATED
7 SELF-EMPLOYED in INCORPORATED
8 Working WITHOUT PAY in family business
9 Unemployed, no work experience in last 5 years (output)
*/

IF qcow='6' or qcow='7' THEN SELFEMP=1;
 ELSE SELFEMP=0;

/* INDUSTRY (QIND) */

/* OCCUPATIONS (QOCC) */

/* ****************** VARIABLES SELECTION: INCLUDE ALL **********************/

/******************* LABELS ********************/
label  AGE = 'Age on the 2000 Census day (defined from agelong)'
 YOB = 'Year of birth (defined from qdb)'
 GQ =  'Whether in a group quarter (defined from rt)'
 POB  = 'State of Birth (defined from qpobst)'
 SOR = 'State of Residence(defined from puid)'
 WHITE = 'white dummy (defined from qrace1)'
 BLACK = 'BLACK DUMMY (DEFEIND FROM QRACE1)'
 WHITEI= 'WHITE DUMMY (DEFINED FROM IMPRACE)'
 BLACKI= 'BLACK DUMMY (DEFINED FROM IMPRACE)'
 HSGRAD = 'HIGH SCHOOL GRADUATE (DEFINED FROM qhigh)'
 SOMCOL = 'SOME COLLEGE (DEFINED FROM qhigh)'
 COLGRA = 'COLLEGE GRADUATE (DEFINED FROM qhigh)' 
 MARRIEDP = 'CURRENTLY MARRIED WITH SPOUSE PRESENT (DEFINED FROM msp)'
 MARRIEDA = 'CURRENTLY MARRIED WITH SPOUSE ABSENT (DEFINED FROM msp)'
 DIVORCED = 'CURRENTLY DIVORCED (DEFINED FROM msp)'
 WIDOWED ='CURRENTLY WIDOWED (DEFINED FROM msp)'
 SEPARATE= 'CURRENTLY SEPARATED (DEFINED FROM msp)'
 SINGLE='NEVER MARRIED OR CURRENTLY SINGLE (FROM msp)'
 VIETNAM = 'VET DURING VIETNAM ERA (FROM vps)'
 ANYSERV = 'EVER SERVED (FROM vps)'
 VET90X00 = 'VET DURING 1990-2000 (FROM vps)' 
 VET75X90 = 'VET DURING 1975-1990 (FROM vps)'
 VETOTH = 'VET DURING EARLIER ERA (FROM vps)' 
 VET95X00='On Active Duty: April 1995 TO 2000 CENSUS DAY (FROM QMIL1)'
 DISSENSE = 'HAVE LONG LASTING VISION OR HEARING IMPAIRMENT (FROM qsense)'
 DPHYSICAL = 'HAVE LONG LASTING LIMITED MOBILITY (FROM qlmob)' /* "physical dis" */
 DISMENTAL = 'DIFFICULTY IN ABILITY TO PERFORM MENTAL TASK (LEARNING, REMEMBERING) (FROM qabmen)'
 DISSELFCARE = 'DIFFICULTY IN ABILITY TO DRESS, BATHE, MOVE ABOUT AT HOME' 
 DISWORK ='DIFFICUTLY IN ABILITY TO WORK (FROM qabwork)'
 DMOBILITY ='DIFFICULTY IN ABILITY TO GO OUTSIDE HOME ALONE (FROM qabgo)' /* "mobility dis" */
 NLABORF='NOT IN LABOR FORCE (FROM esr)'
 EMPLOYED ='EMPLOYED (FROM esr)'
 UNEMPLOYED ='UNEMPLOYED (FROM esr)'
 SELFEMP ='SELF EMPLOYED (FROM qcow)'
 POV='POVERTY STATUS'
 postvet='post-vietnam service'
 nowmili='now in military'
 nowschl='now in school'
 ;

/********************************************************/
/*  SUMMARY     */
/********************************************************/

proc means data=savdat.vardef4 n mean stddev;
proc freq data=savdat.vardef4;
 tables vps YOB AGE esr;
/* NEXT: SEE selstat4.sas for DATA SELECTION AND DESCRIPTIVE STATISTICS; linking with rsn (see link_rsn1.sas) */
