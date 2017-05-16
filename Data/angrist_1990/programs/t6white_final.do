/* t6white_final.do
by S Chen (revised from Simone's and Brigham's programs: t6_newimpute012308.do and newimput_misc030508.do 
20 Sept 2010
Table 6: A. Years of schooling models, B piecewise model
*/
clear
qui set mem 2000m
qui set matsize 500
qui set more off
cd /rdcprojects/br1/br00487
adopath + ./programs/ado

local outputpath ./programs/analysis/AEJ/t6white_final
local outregfile `outputpath'.txt

log using `outputpath'.log, replace text
local neededvars pwt white vietnam logw_weekly elig yob month sob age age2 exper_vet exper_vet2 educ yoc2 yop yos

use `neededvars' if yob>=1948 & yob<=1952 & white==1 & pwt>0 ///
	using ./data/workdat/define_final.dta, replace
drop white
*gen thinout=uniform()
*keep if thinout<.00001

foreach t in 48 49 51 52 {
  	gen t`t'=(yob==19`t')
}
drop yob 

quiet sum exper_vet [aw=pwt] 
scalar m_exper_vet=r(mean) 

xi 		i.sob i.month 

* models for panel A (years of schooling)
local sA	educ
local xA  	_Isob_* _Imonth_*

* models for panel B (piecewise model)
local sB	yoc2
local xB  	yop yos _Isob_* _Imonth_* /* years of primary/secondary schooling as covariates */
 
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

foreach p in A B { /* panels in table 6 */
 foreach m in 2 1 { /* quadratic or linear */
	display "OLS panel`p' model`m'----------------------------"
	reg logw_weekly `s`p'' `ex`m'' `x`p''  [aw=pwt], robust
	if first==1 {
	    	outreg `s`p'' `ex`m'' using `outregfile', ///
      		title(panel`p' model`m') ///
		ctitle(OLS)  ///
		replace nocons noaster se nonotes nor2 bdec(11)
		scalar first=0
     	} 
	else {
		outreg `s`p'' `ex`m'' using `outregfile', ///
      		title(panel`p' model`m') ///
		ctitle(OLS) ///
		append nocons noaster se nonotes nor2 bdec(11) 
	}
	if `m'==1 {
		display "OLS panel`p' model`m' reduce form vet effect ------"
		lincom  -2*exper_vet 
	}
	else {
		display "OLS panel`p' model`m' experience derivative ------"
		lincom exper_vet+2*m_exper_vet*exper_vet2
		display "OLS panel`p' model`m' reduce form vet effect ------"
		lincom  4*exper_vet2-2*exper_vet-4*m_exper_vet*exper_vet2
	}
	foreach a in age yob {
		display "2sls panel`p' model`m' IV=elig+`a' ---------------------------"
		ivreg2b logw_weekly (`s`p'' `ex`m'' = elig `z`m'_`a'') `x`p'' [aw=pwt], robust
		outreg `s`p'' `ex`m'' using `outregfile', ///
		ctitle(2sls `a') ///
		append nocons noaster se nonotes nor2 bdec(11) 

		if `m'==1 {
			display "2sls panel`p' model1 IV=elig+`a'  reduce form vet effect------"
			lincom  -2*exper_vet 
		}
		else {
			display "2sls panel`p' model`m' IV=elig+`a' experience derivative ------"
			lincom exper_vet+2*m_exper_vet*exper_vet2
			display "2sls panel`p'  IV=elig+`a'  reduce form vet effect------"
			lincom  4*exper_vet2-2*exper_vet-4*m_exper_vet*exper_vet2
		}
	}
	display "LIML panel`p'  model`m' IV=elig+yob ---------------------------"
	ivreg2b logw_weekly (`s`p'' `ex`m'' = elig `z`m'_yob') `x`p'' [aw=pwt], liml
		outreg `s`p'' `ex`m'' using `outregfile', ///
		ctitle(liml yob) ///
		append nocons noaster se nonotes nor2 bdec(11) 
	if `m'==1 {
		display "LIML `s`p''  model`m' reduce form vet effect------------" 
		lincom  -2*exper_vet 
	}
	else {
		display "2sls panel`p' model`m' IV=elig+yob experience derivative ------"
		lincom exper_vet+2*m_exper_vet*exper_vet2
		display "LIML panel`p' model`m' reduce form vet effect------------" 
		lincom  4*exper_vet2-2*exper_vet-4*m_exper_vet*exper_vet2
	}
 }
}

display "--- F stat:  First stage F stat for years of schooling/college - adjusted multivariate---"
foreach p in A B { /* panels in table 6 */
 foreach m in 2 1 { /* quadratic or linear */
	display "panel`p' model`m'"
	foreach a in age yob {
		display "IV=elig+`a'--------------"
		ivreg2  `s`p'' (`ex`m'' =  `z`m'_`a'') `x`p'' [aw=pwt], robust
		predict s_res, res
		reg s_res `z`m'_`a'' `x`p'' [aw=pwt], robust 
		test `z`m'_`a''
		drop s_res
	}
 }
}
log close
