capture log close
/* t5white_final.do
DATE: 13 sept 2010
PROGRAMMER: S Chen
DESCRIPTION: This program generates OLS/2SLS estimates (for whites)for AEJ publication, 
for nonwhites 
resultes for whites are in table 5; see t5white_final.do
*/

clear
set mem 1000m
set matsize 500
set more off
cd /rdcprojects/br1/br00487
adopath + ./programs/ado

local outputpath ./programs/analysis/AEJ/ta3nonwhite_final
local outregfile `outputpath'.txt

log using `outregfile', replace text /*this and the next line are because
outreg is stupid and you can't use the "replace" option unless the file already exists
*/
log close

log using `outputpath'.log, replace text

local labor 	employed unemployed nlaborf selfemp qwklyrhr qwklyrwk qincwg logw_weekly qincse
local other	public_sec federal livesob single married evermar
local outcomes 	`labor' `other' 
local z5  	g5_1  g5_2 g5_3 g5_4 g5_5
local z5x   	`z5' g5_1_48 g5_2_48 g5_3_48 g5_4_48 g5_5_48 /*
 */ g5_1_49 g5_2_49 g5_3_49 g5_4_49 g5_5_49 /*
 */ g5_1_51 g5_2_51 g5_3_51 g5_4_51 g5_5_51 /*
 */ g5_1_52 g5_2_52 g5_3_52 g5_4_52 g5_5_52 /*
 */ g5_1_53 g5_2_53 g5_3_53 g5_4_53 g5_5_53
local neededvars	elig vietnam postvet age `outcomes' pwt white yob sob month `z5x'

use `neededvars' if yob>=1948 & yob<=1952 & white==0 ///
	using ./data/workdat/define_final.dta, replace
*gen thinout=uniform()
*keep if thinout<.001
drop white

*REGRESSIONS
*Covs:
xi i.yob i.sob i.month
local x0 _Iyob_* _Isob_* _Imonth_*

scalar first=1

 foreach t in 1950 1948 { 
  foreach y in `outcomes' {
	display "OLS-----------------------"
	reg `y'  vietnam `x0'  [aw=pwt] if yob>=`t', robust 
	sum `y' [aw=pwt] if e(sample)
	while first==1 {
	      	outreg vietnam using `outregfile', ///
      		title(vet effects men born `t'-1952 (table 3))  ///
		ctitle(`y' OLS )  ///
		replace nocons noaster se nonotes nor2 bdec(11)
		scalar first=0
     	}
	outreg vietnam  using `outregfile', ///
	ctitle(`y' OLS) ///
	append nocons noaster se nonotes nor2 bdec(11) 
		ivreg `y' (vietnam  = elig)  `x0' [aw=pwt] if yob>=`t', robust 
		outreg vietnam using `outregfile', ///
      		ctitle(`y' 2SLS-elig)  ///
		append nocons noaster se nonotes nor2 bdec(11) 

		ivreg `y' (vietnam  = `z5x')  `x0' [aw=pwt] if yob>=`t', robust 
		outreg vietnam using `outregfile', ///
      		ctitle(`y' 2SLS-5zx)  ///
		append nocons noaster se nonotes nor2 bdec(11) 
  }
 }

log close
