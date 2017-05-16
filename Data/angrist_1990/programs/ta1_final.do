capture log close
/* ta1_final.do
DATE: 20 sept, 2010
PROGRAMMER: S Chen
DESCRIPTION: This program generates means for variables for Angrist-Chen's Table A1 (AEJ 2010) 
*/

clear
set mem 5000m
set matsize 500
set more off
cd /rdcprojects/br1/br00487
adopath + ./programs/ado

local outputpath ./programs/analysis/AEJ/ta1_final
local outregfile `outputpath'.txt

log using `outregfile', replace text /*this and the next line are because
outreg is stupid and you can't use the "replace" option unless the file already exists
*/
log close

log using `outputpath'.log, replace text

local A_ed     	educ yoc2 edu_9 edu_10 edu_11 edu_12 edu_hs edu_sc1  edu_sc2 edu_as edu_ba edu_ma edu_pr edu_phd
local B_labor 	employed unemployed nlaborf selfemp qwklyrhr qwklyrwk qincwg logw_weekly qincse Dssi Dss 
local C_other 	public_sec federal livesob single married evermar
local neededvars elig vietnam postvet age `A_ed' `B_labor' `C_other' pwt white yob

use `neededvars' if yob>=1948 & yob<=1952 ///
	using ./data/workdat/define_final.dta, replace
*gen thinout=uniform()
*keep if thinout<.01

local stat  	elig vietnam postvet age `A_ed' `B_labor' `C_other' 
foreach white in 1 0 {
	display "white = `white', all"
	sum `stat' [aw=pwt] if white == `white' & (vietnam==1 | vietnam==0),  separator(0)
	display "white = `white', vets"
	sum `stat' [aw=pwt] if white == `white' & vietnam==1,  separator(0)
	display "white = `white', non-vets"
	sum `stat' [aw=pwt] if white == `white' & vietnam==0,  separator(0)
}
log close
