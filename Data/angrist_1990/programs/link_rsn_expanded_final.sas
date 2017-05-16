/* 
1/9/2007, 4/07/2007 including older cohorts (1946-1955) 
4/15/07
march 2008 : incl. cohorts born between 1944-55
9 Sept 2010: use the corrected version of the rsn file  
Goal: Link RSN (dobrsn20091215.csv) with Census 2000 (vardef4)
background: 
 data construction- using getdata4.sas to extract variables from decenial 2000
 var definitions- using vardef4.sas
 sample selection and desc stat- see below
 rsn (dobrsn20091215.csv) - from josh; corrected by brigham  
*/
libname dat 	'/rdcprojects/br1/br00487/data/workdat';
libname lottery '/rdcprojects/br1/br00487/data/lottery';

options linesize=120;  

/****************************************/
/* SAMPLE SELECTION OF CENSUS  */

data census1x;
 set dat.vardef4;
 if SOR>=1 & SOR<=56;
 if POB>=1 & POB<=56;
 if YOB>=1944 & YOB<=1955;
 /* GENERATE DATE OF BIRTH FROM qdb*/
        year=yob-1900;
 month=substr(qdb,5,2)/1.;
 day=substr(qdb,7,2)/1.;
 LABEL  year ='YEAR OF BIRTH'
  month ='MONTH OF BIRTH'
  day='DAY OF BIRTH'; 
proc means n mean min max;
title 'census extract';

/********************************/
/* LOAD LOTTERY NUMBERS  */

PROC IMPORT OUT= lottery1x 
            DATAFILE= "/rdcprojects/br1/br00487/data/lottery/dobrsn20091215.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 

proc means data=lottery1x n mean min max;
title 'lottery numbers before match';

/******************************************************/
/* LINK LOTTERY NUMBER WITH CENSUS 2000 BY BIRTH DATE */

proc sort data=census1x; by year month day;
proc sort data=lottery1x; by year month day;

data dat.vardef4x_rsn_final;
 MERGE census1x lottery1x;
 BY year month day;
 LABEL year ='YEAR OF BIRTH'
  month ='MONTH OF BIRTH'
  day='DAY OF BIRTH'
  rsn='LOTTERY NUMBER';

proc means data=dat.vardef4x_rsn_final n mean min max;
title 'matched census/lottery extract';

/*
proc print data=dat.vardef4x_rsn_final;
where pob eq .;
var year month day rsn;
title 'bad bdays';

proc print data=dat.vardef4x_rsn_final;
where rsn eq .;
var year month day pob;
title 'missing rsn';
proc freq data=dat.vardef4x_rsn;
 tables rsn;
 title 'f(rsn)';
*/
/* create a temp extract in the working directory. only include the variables we need for the project */ 
data work.temp;
	set dat.vardef4x_rsn_final;
	keep AGE ANYSERV COLGRA DIVORCED EMPLOYED GQ HGC HSGRAD MARRIEDA MARRIEDP NLABORF POB SELFEMP SEPARATE SINGLE /*
	*/ SOMCOL SOR UNEMPLOYED VIETNAM WHITE WIDOWED YOB day disable esr month msp postvet prt pwt qcitizen qcow qdb /*
	*/ qgrade qhigh qincint qincoth qincpa qincret qincse qincss qincssi qinctot qincwg qwklyrwk rsn rt vps year;                      

/* export to a STATA file  */
proc export data=work.temp
 outfile="/rdcprojects/br1/br00487/data/workdat/final.dta"
 dbms = stata replace;
