/* t6ftest.do
by S Chen
17 Sept 2010
*/
clear
qui set mem 2000m
qui set matsize 500
qui set more off
cd /rdcprojects/br1/br00487
qui do ./programs/ado/i/ivreg2.ado
*adopath + ./programs/i/ado

local outputpath ./programs/analysis/AEJ/t6ftest
local outregfile `outputpath'.txt

log using `outputpath'.log, replace text
local neededvars pwt white vietnam logw_weekly elig yob month sob age age2 exper_vet exper_vet2 educ yoc2 

use `neededvars' if yob>=1948 & yob<=1952 & white==1 & pwt>0 ///
	using ./data/workdat/define_final.dta, replace
drop white
gen thinout=uniform()
keep if thinout<.0001

foreach t in 48 49 51 52 {
  gen t`t'=(yob==19`t')
}
drop yob 
sum t4* t5*

quiet sum exper_vet [aw=pwt]
scalar m_exper_vet=r(mean)

*years of primary school
gen yop=min(educ,8)

* years of secondary school
gen yosec=min(max(educ-8,0),4)

xi 		i.sob i.month 

* models for panel A (years of schooling)
local sA	educ
local xA  	_Isob_* _Imonth_*

* models for panel B (years of college)
local sB	yoc2
local xB  	yop yosec _Isob_* _Imonth_* 
 
* endog covs:
local ex1	exper_vet /* linear model*/
local ex2  	exper_vet exper_vet2 /* quadratic model */

* instruments in addition to elig: 
* linear model 
local z1_age	age 
local z1_yob	t48 t49 t51 t52

* quadratic model
local z2_age	age age2
local z2_yob	t48 t49 t51 t52

scalar first=1

foreach p in A B { 
 foreach m in 1 2 { /* linear or quadratic */
	display "OLS panel`p' model`m'----------------------------"

/* covariates */
local y  logw_weekly
local X1 exper_vet
local X2 exper_vet2
local X  `X1' `X2'
local W  i.sob i.month 
local Z  elig age age2

* First stage F-stat, linear experience effects model
	/* === 1. construct first stage fitted values for X from regressions of X on W and Z  === */
	xi: reg `X1' `Z' `W' [pw=pwt], robust 
	predict X1_hat if e(sample)

	/*=== 2. regress S on X_hat and W */
	xi: reg educ X1_hat `W' [pw=pwt], robust
	predict S_r if e(sample), res

	/* 3. regress y on S_r by 2SLS using Z and W as instruments (check this is usual 2SLS estimates for schooling) */
	xi: ivreg logw_weekly (S_r= `Z') `W' [pw=pwt], robust

	/* 4. regress S_r on W and Z */
	xi: reg S_r `Z' `W'  [pw=pwt], robust
	test `Z'
	drop S_r
	drop X1_hat 
log close
