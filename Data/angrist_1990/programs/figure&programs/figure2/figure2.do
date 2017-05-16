***************************************
* PROGRAM: figure2.do
* PROGRAMMER: Brigham Frandsen (S Chen)
* PURPOSE: Update figure2
* DATE CREATED: 2/25/2010 (3/22/2010) (10 sept 2010)
***************************************
capture log close
log using "C:\Research\Published\AngristChenAEJ\paper\final\programs\figure2\figure2.log", replace text
clear
set mem 500m
set more off


***********************************************************************
* FIGURE 3
***********************************************************************
	* Read in excel data
clear
insheet using "C:\Research\Published\AngristChenAEJ\paper\final\programs\figure2\Brigham_Graphnew_SC2010March.csv", names
label define rank 1 "7th-8th" 2 "9th" 3 "10th" 4 "11th" 5 "12th (no diploma)" ///
	6 "High school graduate" 7 "Some college (<1 year)" 8 "Some college (>=1 year)" ///
	9 "Associate's degree" 10 "B.A." 11 "M.A." 12 "Professional Degree" ///
	13 "Ph.D."
label values rank rank

foreach race in 0 1 {
sum coeff if white==`race' & byear5_52==0 & rank>=1
scalar maxcoeff=r(max)
scalar mcoeff=r(mean)

}
sort rank

	
*Graphs with histogram for vets only
* WHITES
graph twoway ///
 (line coeff upper lower rank if white & byear5_52==0 & rank>=1, ///
 clp(solid dash dash) clw(medthick medthick medthick) ylabel(0(.025).1,axis()) yaxis() ytitle("Estimates") ///
 clc(black gs10 gs10)) /// 
 , name(BF_4852wvet, replace)  ///
 legend(off) xlabel(1(1)13, valuelabel angle(45)) yline(0) ///
 xtitle("") 
 
graph export figure2.eps, as(eps) replace

log close
