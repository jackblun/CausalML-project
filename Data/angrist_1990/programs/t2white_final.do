/* t2white_final.do: tabe 2 first stage, whites 
for AEJ the final step for publication
*/

set mem 1000m
set matsize 500
set more off
cd /rdcprojects/br1/br00487
adopath + ./programs/ado

local outputpath ./programs/analysis/AEJ/t2white_final
local outregfile `outputpath'.txt

log using `outregfile', replace text /*this and the next line are because
outreg is stupid and you can't use the "replace" option unless the file already exists
*/
log close

log using `outputpath'.log, replace text

local covs sob yob month pwt white 
local z5  g5_1  g5_2 g5_3 g5_4 g5_5
local z5  g5_1  g5_2 g5_3 g5_4 g5_5
local z5x `z5' g5_1_48 g5_2_48 g5_3_48 g5_4_48 g5_5_48 /*
 */ g5_1_49 g5_2_49 g5_3_49 g5_4_49 g5_5_49 /*
 */ g5_1_51 g5_2_51 g5_3_51 g5_4_51 g5_5_51 /*
 */ g5_1_52 g5_2_52 g5_3_52 g5_4_52 g5_5_52 /*
 */ g5_1_53 g5_2_53 g5_3_53 g5_4_53 g5_5_53
local neededvars  vietnam elig `covs' `z5' `z5x'

use `neededvars' if yob>=1948 & yob<=1953 & white==1 & pwt>0 ///
	using ./data/workdat/define_final.dta, replace
drop white

*gen thinout=uniform()
*keep if thinout<.01

* covariates 
local x0  i.sob i.month i.yob 

*REGRESSIONS
* (Row 1) IV = elig ---------------------------
scalar first=1

foreach t in 50 48 {
  display "-------First stage (elig) 19`t'-52-----------------"
  xi: regress vietnam elig `x0' [aw=pwt] if yob>=19`t' & yob<=1952, robust
	if first==1 {
	    	outreg elig using `outregfile', ///
      		title(t2 first stage(elig)) ///
		ctitle(19`t'-52)  ///
		replace nocons noaster se nonotes nor2 bdec(11)
		scalar first=0
     	} 
	else {
		outreg elig using `outregfile', ///
		ctitle(19`t'-52) ///
		append nocons noaster se nonotes nor2 bdec(11) 
	}
}

foreach t in 48 49 50 51 52 53 {
  display "-------First stage (elig) 19`t'-----------------"
  xi: regress vietnam elig `x0' [aw=pwt] if yob==19`t', robust
		outreg elig using `outregfile', ///
		ctitle(19`t') ///
		append nocons noaster se nonotes nor2 bdec(11) 
}

* (Rows 2-6 Cols 1-2) 5z -----------------

foreach t in 50 48 {
  display "-------First stage (z5) 19`t'-52-----------------"
  xi: regress vietnam `z5' `x0' [aw=pwt] if yob>=19`t' & yob<=1952, robust
  display "-------F-test (z5) 19`t'-52-----------------"
  test `z5'
		outreg `z5' using `outregfile', ///
      		title(t2 first stage(z5)) ///
		ctitle(19`t'-52) ///
		append nocons noaster se nonotes nor2 bdec(11) 
}

* (Rows 2-6 Cols 3-) 5zx -----------------

foreach t in 48 49 50 51 52 53 {
  display "-------First stage (z5x) 19`t'-----------------"
  xi: regress vietnam `z5' `x0' [aw=pwt] if yob==19`t', robust
  display "-------F-test (z5x) 19`t'-----------------"
  test `z5'
		outreg `z5' using `outregfile', ///
      		title(t2 first stage(z5))  ///
		ctitle(19`t') ///
		append nocons noaster se nonotes nor2 bdec(11) 
}
log close


