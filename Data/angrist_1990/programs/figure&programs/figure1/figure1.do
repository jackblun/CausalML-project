***************************************************************
* PROGRAM: fig1trial.do
* PROGRAMMER: Brigham Frandsen
* PURPOSE: Makes figure 1 from group means from spreadsheet
* cellavgstograph.csv. It does smoothing within eligibility groups.
* DATE CREATED: 2-25-2010 
* LAST MODIFIED:
****************************************************************
clear

local cutoff1950 195
local cutoff1951 125
local cutoff1952 95
local cutoff1953 95
insheet using "C:\Research\Published\AngristChenAEJ\paper\final\programs\figure1\cellavgstograph.csv"
*Define elibility groups

foreach j in 1950 1951 1952 1953 {
  gen elig`j' = 0
  replace elig`j' = 1 if rsn<=`cutoff`j'' 
 }
* LOWESS PLOT OF P(VETERAN|RSN) BY RACE AND YEAR OF BIRTH
* FIRST DO FOR EACH OF 8 GROUPS (THIS WILL TAKE FOREVER)

sort rsn

local bw 4
foreach i in 0 1 {/*loop over race*/
 foreach j in 1950 1951 1952 1953 {/*loop over yob*/
	gen p_vet`i'`j' = 999
	foreach k in 0 1 {/*loop over elig/inelig*/
		lowess vet`i'`j' rsn if elig`j'==`k', mean nograph  generate(p_vet`i'`j'`k') bwidth(.`bw')
		replace p_vet`i'`j'=p_vet`i'`j'`k' if elig`j'==`k'
		
		drop p_vet`i'`j'`k'
 }
 label var p_vet`i'`j' "`j'"
}
}

* MAKE INDIVIDUAL GRAPHS
line p_vet1* rsn, scheme(s1mono) name(white2, replace) ///
 ysc(r(.05 .5) titleg(1.5)) ///
 title("A. Whites", size(medsmall)) ytitle("P(Veteran|RSN)") xtitle(RSN) ///
 ylabel(.05(.1).5,nogrid) xlabel(1(364)365 50 100 150 200 250 300) xsca(titlegap(1.5)) nodraw ///
 text(`end1950_1' 380   "1950", just(left) size(small)) ///
 text(`end1951_1' 380 "1951", just(left) size(small)) ///
 text(`end1952_1' 380 "1952", just(left) size(small)) ///
 text(`end1953_1' 380 "1953", just(left) size(small)) ///
 clp(solid dash longdash shortdash) clw(medthick medthick medthick medthick ) clcolor(black black black black) ///
 legend(subtitle("Year of Birth"))

line p_vet0* rsn, scheme(s1mono) name(nonwhite2, replace) ///
 ysc(r(.05 .5) titleg(1.5)) ///
 title("B. Nonwhites", size(medsmall)) ytitle("P(Veteran|RSN)") xtitle(RSN) ///
 ylabel(.05(.1).5,nogrid) xlabel(1(364)365 50 100 150 200 250 300) xsca(titlegap(1.5)) nodraw ///
 text(`end1950_0' 380   "1950", just(left) size(small)) ///
 text(`end1951_0' 380 "1951", just(left) size(small)) ///
 text(`end1952_0' 380 "1952", just(left) size(small)) ///
 text(`end1953_0' 380 "1953", just(left) size(small)) ///
 clp(solid dash longdash shortdash) clw(medthick medthick medthick medthick ) clcolor(black black black black) ///
 legend(subtitle("Year of Birth"))

* TWO PANEL GRAPH
gr combine white2 nonwhite2, scheme(s1mono) rows(2) col(1) ysize(9) xsize(6) saving(pvet_seg_smooth_`bw', replace) ///
/* title("Figure 1. First-stage plots. The relation between probability of military" ///
  "service and draft lottery numbers\n" ///
  "Note: Lowess smoothing with a bandwidth of .`bw' is used within eligibility" ///
  "groups", pos(7) size(small)) */
 graph export "C:\Research\Published\AngristChenAEJ\paper\final\programs\figure1\figurel.eps", replace 

* YOU CAN COMMENT THIS OUT IF YOU WANT TO PLAY WITH GRAPHS


*******************************************************************************************
* PROGRAM DESCRIPTION: This program will make Fig 2 graphs (P(Vet|RSN) by race and birth 
* year) and save the graph filesand Fig2_1Stage to your current directory. Doing this file
* will upload the program into STATA
* 
* 
* If you have questions, email me at frandsen@mit.edu
***************************************************************************************



