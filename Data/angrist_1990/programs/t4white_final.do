/* t4white_final.do
DATE: 13 Sept 2010
PROGRAMMER: S Chen
DESCRIPTION: This program generates OLS/2SLS estimates (for both whites and nonwhites)for AEJ revision
results for white in table 4
results for nonwhites are in appendix (see ta2nonwhite_final.do)
*/

set mem 1000m
set matsize 500
set more off
cd /rdcprojects/br1/br00487
adopath + ./programs/ado

local outputpath ./programs/analysis/AEJ/t4white_final
local outregfile `outputpath'.txt

log using `outregfile', replace text /*this and the next line are because
outreg is stupid and you can't use the "replace" option unless the file already exists
*/
log close

log using `outputpath'.log, replace text

local ed     	educ yoc2 edu_sc1  edu_sc2 edu_as edu_ba edu_ma 
local z5  	g5_1  g5_2 g5_3 g5_4 g5_5

local neededvars	elig vietnam `ed' pwt white yob sob month `z5'

use `neededvars' if yob>=1948 & yob<=1952 & white==1 ///
	using ./data/workdat/define_final.dta, replace
gen thinout=uniform()
*keep if thinout<.001
*drop white

*REGRESSIONS
*Covs:
local x0 i.yob i.sob i.month

scalar first=1

 foreach t in 1948 1949 1950 1951 1952 { 
  foreach y in `ed' {
	display "5z 2SLS-----------------------"
	xi: ivreg `y' (vietnam  = `z5')  `x0' [aw=pwt] if yob==`t', robust 
	if first==1 {
      		outreg vietnam using `outregfile', ///
		title(`t')  ///
		ctitle(`y') replace nocons noaster se nonotes nor2 bdec(11)
		scalar first=0
     	}
	else {
	outreg vietnam using `outregfile', ///
	ctitle(`y') append nocons noaster se nonotes nor2 bdec(11) 
	}
  }
 }

log close
