//file: merge institution data
//merge in together all of the settler mortality instrument datasets from paper
//data source: https://economics.mit.edu/faculty/acemoglu/data/ajr2001
global datadir "/media/luis/hdd3/papers/HistoricalEcon_W_ML/Acemoglu_SettlerMortality"
set more off
local t=1
forval t=1/8{
	use ${datadir}/maketable`t', clear
	desc
}
use ${datadir}/maketable4, clear
tempfile panel
save `panel', replace
foreach t in 5 6 7 8 2 1 {
	di "Table `t' Data" _n
	use ${datadir}/maketable`t', clear
	if `t'==1{
		di "Dropping Dups in T1 dta-----"
		drop if mi(shortnam)
		egen missdata = rowmiss(_all)
		bys shortnam (missdata): keep if _n==1
		drop missdata
	}
	tempfile usedata
	save `usedata',replace
	di "Merging Dups-----"	
	use `panel', clear
	merge 1:1 shortnam using `usedata', gen(table`t'_merge) update
	drop if table`t'_merge==1 | table`t'_merge==2
	drop table`t'_merge
	save `panel', replace
}
label var oilres "Oil Resources (Barrels/Capita)"
//create missing dummies
foreach v of varlist africa-extmort4{
	//skip indicators of a group (only get mi var for one of them
	if "`: var label `v''"=="" continue
	gen mi_`v' = mi(`v')
	su mi_`v', meanonly
	if `r(sum)'==0 {
		drop mi_`v'
	}
	else{
		label var mi_`v' "Missing Indicator for `v'"
		di "`r(sum)' obs missing `v'"
	}
}

//export variables

fsum _all, s(n mean) label

export delimited using ${datadir}/colonial_origins_data.csv, replace
//change missings to zero; create alt dataset
foreach v of varlist _all{
	if substr("`v'",1,4)=="temp"{
		replace `v'=0 if mi_temp1
	}
	else if strpos("`v'","humid"){
		replace `v'=0 if mi_humid1 
	}
	else if inlist("`v'","steplow","deslow","stepmid","desmid","drystep","drywint"){
		replace `v' = 0 if mi_steplow
	}
	else if inlist("`v'","goldm","iron","silv","zinc"){
		replace `v' = 0 if mi_goldm
	}
	else{
		cap confirm var mi_`v'
		if _rc	continue
		else {
			replace `v' = 0 if mi_`v'
		}
	}
	
}
export delimited using ${datadir}/colonial_origins_data_missimp.csv, replace
