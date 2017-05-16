/* 
output: define_final.dta
date: 21 sept 2010
by: s chen
define variables before analysis
update year of schooling imputation using Jaeger's and Frandsen's imputation 
define year of primary/seconday education
*/
clear
set mem 5000m
cd /rdcprojects/br1/br00487
log using ./programs/analysis/define/define_final, replace text

use ./data/workdat/final.dta, clear

/* define the eligibility dummy (elig) */
gen elig=0
replace	elig=1 if (year>=44 & year<=50) & (rsn <= 195)
replace elig=1 if year==51 & (rsn <= 125)
replace elig=1 if (year==52 | year==53) & (rsn <= 95)

foreach t in 47 48 49   51 52 53 {
 gen elig_`t'=elig*(year==`t')
}

/* define other outcome variables */
gen married=(marriedp==1 | marrieda==1) /* currently married */
gen evermar=(married==1 | divorced==1 | separate==1 | widowed==1) /* ever married */
gen livesob=(pob==sor) /* live in state of birth */
/*
QCOW	Class of Worker	0	Not in universe (less than 16 years old or did not work in the last 5 years)					1	QCOW	Class of Worker	1		Employee of PRIVATE FOR PROFIT
			2		Employee of PRIVATE NOT-FOR-PROFIT
			3		Employee of LOCAL GOVERNMENT
			4		Employee of STATE GOVERNMENT
			5		Employee of FEDERAL GOVERNMENT
			6		SELF-EMPLOYED in NOT INCORPORATED
			7		SELF-EMPLOYED in INCORPORATED
			8		Working WITHOUT PAY in family business
			9		Unemployed, no work experience in last 5 years (output)
*/
gen non_profit=(real(qcow)==2) /* work in non profit */
gen public_sec=(real(qcow)>=3 & real(qcow)<=5) /* work in public sector */
gen federal=(real(qcow)==5) /* work in federal govt */

/* earnings variables and dummies */
gen Dother=(qincoth>0) /* other income (mostly VDC)>0 */
gen Dss=(qincss>0) /* Social security income excluding SSI (mostly SSDI)>0 */ 
gen Dssi=(qincssi>0) /* Supplemental security income (SSID)>0 */
gen Dret=(qincret>0) /* retirement income >0 */
gen Dwage=(qincwg>0) /* wage>0 */
gen Dse=(qincse>0)  /* self employment income */
gen qtotxfer=qincoth + qincss + qincssi /* total federal transfer income */
gen Dxfer=qtotxfer>0 /* total transfer >0 */

gen w_weekly=(qincwg/qwklyrwk) if qwklyrwk>0 /* weekly earnings */
gen logw_weekly=ln(w_weekly)  /* ln(weekly earnings) */

/* 5 instruments */
gen g5_1=(rsn>=1 & rsn<=95)
gen g5_2=(rsn>=96 & rsn<=125)
gen g5_3=(rsn>=126 & rsn<=160)
gen g5_4=(rsn>=161 & rsn<=195)
gen g5_5=(rsn>=196 & rsn<=230)
foreach g in 1 2 3 4 5{
 foreach t in 47 48 49   51 52 53 {
  gen g5_`g'_`t'=g5_`g'*(year==`t')
 }
}

/* 
EDUCATION categories (qhigh) 
00  Not in universe (less than 3 years old)
01  No schooling completed
02  Nursery school to 4th grade (2.5)
03  5th grade or 6th grade (5.5)
04  7th grade or 8th grade (7.5)
05  9th grade
06  10th grade
07  11th grade
08  12th grade, no diploma
09  High school graduate (12)
10  Some college, but less than 1 year (13)
11  1 or more years of college, no degree (13)
12  Associate degree (14)
13  Bachelor's degree (16)
14  Master's degree (18)
15  Professional degree (18)
16  Doctorate degree (18)
*/
/* years of schooling; see Jaeger 1997 and Frandsen's imputation; we use median */
gen educ=0 if real(qhigh)==1
replace educ=3 if real(qhigh)==2
replace educ=6 if real(qhigh)==3
replace educ=8 if real(qhigh)==4
replace educ=9 if real(qhigh)==5
replace educ=10 if real(qhigh)==6
replace educ=11 if real(qhigh)==7
replace educ=11.38 if real(qhigh)==8
replace educ=12 if real(qhigh)==9
replace educ=12.557415 if real(qhigh)==10
replace educ=13.35 if real(qhigh)==11
replace educ=14 if real(qhigh)==12
replace educ=16 if real(qhigh)==13
replace educ=18 if real(qhigh)>=14 & real(qhigh)<=16

gen edu_78=real(qhigh)>=4 & real(qhigh)<=16
gen edu_9=real(qhigh)>=5 & real(qhigh)<=16
gen edu_10=real(qhigh)>=6 & real(qhigh)<=16
gen edu_11=real(qhigh)>=7 & real(qhigh)<=16
gen edu_12=real(qhigh)>=8 & real(qhigh)<=16
gen edu_hs=real(qhigh)>=9 & real(qhigh)<=16
gen edu_sc1=real(qhigh)>=10 & real(qhigh)<=16
gen edu_sc2=real(qhigh)>=11 & real(qhigh)<=16
gen edu_as=real(qhigh)>=12 & real(qhigh)<=16
gen edu_ba=real(qhigh)>=13 & real(qhigh)<=16
gen edu_ma=real(qhigh)>=14 & real(qhigh)<=16
gen edu_pr=real(qhigh)>=15 & real(qhigh)<=16
gen edu_phd=real(qhigh)==16

*Generate schooling categorization
*less than HS grad
gen educCAT=0

*hs grad:
replace educCAT=1 if real(qhigh)==9

*some college, no degree
replace educCAT=2 if real(qhigh)>=10 & real(qhigh)<12

*college degree (could be two-year)
replace educCAT=4 if real(qhigh)>=12

tab educCAT

/*GENERATE PIECEWISE EDUCATION VARIABLES */
*Years of Primary:
gen yop=min(educ,8)
label var yop "years of primary school"

/*Years of Secondary */
gen yos=min(max(educ-8,0),4)
label var yos "years of secondary school"

/*Years of College */
gen yoc2=min(max(educ-12,0),4)
label var yoc2 "years of college"

gen n_yoc=educ-yoc2

*Check:
gen check=yop+yos+yoc-min(educ,16)
sum check
if ~(r(min)==r(max)&r(min)==0) display "******Piecewise education breakup failed******"

/* work experience */
/* note that SAS automatically convert all varialbe names (e.g., AGE) in capital to those in lower case (age) */
gen exper =age-educ-6
gen exper_vet =exper-2*vietnam
gen exper_veta =age-18-2*vietnam 

gen exper_vetm=.
foreach w in 1 0 {
foreach t in 47 48 49 50 51 52 53 {
 sum educ [aw=pwt] if year==`t' & white==`w'
 replace exper_vetm=age-r(mean)-6-2*vietnam if year==`t' & white==`w'
}
}
gen exper2=exper^2
gen exper_vet2=exper_vet^2
gen exper_veta2=exper_veta^2
gen exper_vetm2=exper_vetm^2
gen age2=age^2

gen not_working = 1-employed

/* redefine postvet (5-11-07) */
replace postvet=real(vps)>=1 & real(vps)<=7

/* rename */
rename pob sob 
compress
describe
save ./data/workdat/define_final.dta, replace

log close


