/* 1/3/2007 Angrist-Chen's project on Vet; 
* data extraction from decennial 2000 (sedf**03 and 05)
* Include  (i) males
* note: state of residence (SOR)= the first 2 digits of PUID
* SAVE AS SAS DATA
*/

libname dec '/decennial/2000cen/microdata'; 
libname Mylib '/rdcprojects/br00487/data/extract';

data Mylib.usmen01032007;

set dec.sedfak03 dec.sedfal03 dec.sedfar03 dec.sedfaz03 dec.sedfca03 /* 
*/ dec.sedfco03 dec.sedfct03 dec.sedfdc03 dec.sedfde03 dec.sedffl03 /* 
*/ dec.sedfga03 dec.sedfhi03 dec.sedfia03 dec.sedfid03 dec.sedfil03 /* 
*/ dec.sedfin03 dec.sedfks03 dec.sedfky03 dec.sedfla03 dec.sedfma03 /* 
*/ dec.sedfmd03 dec.sedfme03 dec.sedfmi03 dec.sedfmn03 dec.sedfmo03 /* 
*/ dec.sedfms03 dec.sedfmt03 dec.sedfnc03 dec.sedfnd03 dec.sedfne03 /* 
*/ dec.sedfnh03 dec.sedfnj03 dec.sedfnm03 dec.sedfnv03 dec.sedfny03 /* 
*/ dec.sedfoh03 dec.sedfok03 dec.sedfor03 dec.sedfpa03 dec.sedfri03 /* 
*/ dec.sedfsc03 dec.sedfsd03 dec.sedftn03 dec.sedftx03 dec.sedfut03 /* 
*/ dec.sedfva03 dec.sedfvt03 dec.sedfwa03 dec.sedfwi03 dec.sedfwv03 /* 
*/ dec.sedfwy03 dec.sedfpr03 /* 
*/ dec.sedfak05 dec.sedfal05 dec.sedfar05 dec.sedfaz05 dec.sedfca05 /* 
*/ dec.sedfco05 dec.sedfct05 dec.sedfdc05 dec.sedfde05 dec.sedffl05 /* 
*/ dec.sedfga05 dec.sedfhi05 dec.sedfia05 dec.sedfid05 dec.sedfil05 /* 
*/ dec.sedfin05 dec.sedfks05 dec.sedfky05 dec.sedfla05 dec.sedfma05 /* 
*/ dec.sedfmd05 dec.sedfme05 dec.sedfmi05 dec.sedfmn05 dec.sedfmo05 /* 
*/ dec.sedfms05 dec.sedfmt05 dec.sedfnc05 dec.sedfnd05 dec.sedfne05 /* 
*/ dec.sedfnh05 dec.sedfnj05 dec.sedfnm05 dec.sedfnv05 dec.sedfny05 /* 
*/ dec.sedfoh05 dec.sedfok05 dec.sedfor05 dec.sedfpa05 dec.sedfri05 /* 
*/ dec.sedfsc05 dec.sedfsd05 dec.sedftn05 dec.sedftx05 dec.sedfut05 /* 
*/ dec.sedfva05 dec.sedfvt05 dec.sedfwa05 dec.sedfwi05 dec.sedfwv05 /* 
*/ dec.sedfwy05 dec.sedfpr05;

keep PUID PWT AGELONG RT PRT QREL QSPAN QDB QSEX QRACE1 QRACE2 IMPRACE QMS MSP QATTEND QGRADE QHIGH QANCESCODE1 /* 
*/ QANCESCODE2 QSPEAK QLANGCODE QENGABIL QPOBST QCITIZEN QYR2US  QINCTOT QINCWG QINCSE QINCINT QINCSS /* 
*/ QINCSSI QINCPA QINCRET QINCOTH QWKLYRWK QWKLYRHR VPS DISABLE QSENSE QLMOB QABMEN QABPHYS QABWORK /*
*/ QABGO ESR QLAYOFF QLOOKWK QCOW QIND QOCC POV QMIL1 QMILAD QMILTOT;

/* SAMPLE SELECTION */
/* MALE ONLY */
if QSEX='2' then delete;




