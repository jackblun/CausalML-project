/* t7white_final.do
16 Sept 2010
by S Chen
*/
clear
set mem 1000m
set matsize 500
set more off
cd /rdcprojects/br1/br00487
adopath + ./programs/ado

local outputpath ./programs/analysis/AEJ/t7white_final
local outregfile `outputpath'.txt

log using `outregfile', replace text /*this and the next line are because
outreg is stupid and you can't use the "replace" option unless the file already exists
*/
log close

log using `outputpath'.log, replace text

local work   	public_sec federal 
local marry	single married evermar
local outcomes 	`work' livesob  `marry'
local z5  	g5_1  g5_2 g5_3 g5_4 g5_5
local z5x   	`z5' g5_1_48 g5_2_48 g5_3_48 g5_4_48 g5_5_48 /*
 */ g5_1_49 g5_2_49 g5_3_49 g5_4_49 g5_5_49 /*
 */ g5_1_51 g5_2_51 g5_3_51 g5_4_51 g5_5_51 /*
 */ g5_1_52 g5_2_52 g5_3_52 g5_4_52 g5_5_52 /*
 */ g5_1_53 g5_2_53 g5_3_53 g5_4_53 g5_5_53
local neededvars	elig vietnam postvet age `outcomes' `z5x' pwt white yob sob month 

use `neededvars' if yob>=1948 & yob<=1952 & white==1 ///
	using ./data/workdat/four.dta, replace
*gen thinout=uniform()
*keep if thinout<.001

/***************/
/* regressions */
*Covs:
local x0 i.yob i.sob i.month

scalar first=1

 foreach t in 1950 1948 { 
	display "stat stat `t'-1952--------------------"
  	sum `outcomes' [aw=pwt] if yob>=`t'
 }

 foreach t in 1950 1948 { 
  foreach y in `outcomes' {
	display "OLS-----------------------"
	xi: reg `y'  vietnam `x0'  [aw=pwt] if yob>=`t', robust 
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
		xi: ivreg `y' (vietnam  = elig)  `x0' [aw=pwt] if yob>=`t', robust 
		outreg vietnam using `outregfile', ///
      		ctitle(`y' 2SLS-elig)  ///
		append nocons noaster se nonotes nor2 bdec(11) 

		xi: ivreg `y' (vietnam  = `z5x')  `x0' [aw=pwt] if yob>=`t', robust 
		outreg vietnam using `outregfile', ///
      		ctitle(`y' 2SLS-5zx)  ///
		append nocons noaster se nonotes nor2 bdec(11) 
  }
 }

log close



