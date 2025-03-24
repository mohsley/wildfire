@echo off
setLocal EnableDelayedExpansion

REM -----------------------------------------------

set DIR=c:
set PGM=%DIR%\hysplit
cd %PGM%\working

REM Create output directories if they don't exist
mkdir conc_output 2>nul
mkdir traj_output 2>nul

REM -----------------------------------------------

IF EXIST ASCDATA.CFG DEL ASCDATA.CFG
echo -90.0   -180.0  lat/lon of lower left corner   >ASCDATA.CFG
echo 1.0     1.0     lat/lon spacing in degrees    >>ASCDATA.CFG
echo 180     360     lat/lon number of data points >>ASCDATA.CFG
echo 2               default land use category     >>ASCDATA.CFG
echo 0.2             default roughness length (m)  >>ASCDATA.CFG
echo '%PGM%\bdyfiles\'  directory of files         >>ASCDATA.CFG

REM -----------------------------------------------

echo 25 01 08 15             >CONTROL
echo 1                      >>CONTROL
echo 34.03 -118.33 500      >>CONTROL
echo 24                     >>CONTROL
echo 0                      >>CONTROL
echo 15500                  >>CONTROL
echo 1                      >>CONTROL
echo C:/hysplit/HYSPLIT/    >>CONTROL
echo gdas1.jan25.w2         >>CONTROL
echo 1                      >>CONTROL
echo TEST                   >>CONTROL
echo 1.0                    >>CONTROL
echo 1.0                    >>CONTROL
echo 00 00 00 00 00         >>CONTROL
echo 1                      >>CONTROL
echo 0.0 0.0                >>CONTROL
echo 0.05 0.05              >>CONTROL
echo 30.0 30.0              >>CONTROL
echo ./                     >>CONTROL
echo cdump                  >>CONTROL
echo 1                      >>CONTROL
echo 100                    >>CONTROL
echo 00 00 00 00 00         >>CONTROL
echo 00 24 00 00 00         >>CONTROL
echo 00 01 00               >>CONTROL
echo 1                      >>CONTROL
echo 0.0 0.0 0.0            >>CONTROL
echo 0.0 0.0 0.0 0.0 0.0    >>CONTROL
echo 0.0 0.0 0.0            >>CONTROL
echo 0.0                    >>CONTROL
echo 0.0                    >>CONTROL

REM -----------------------------------------------

IF EXIST cdump DEL cdump
IF EXIST SETUP.CFG DEL SETUP.CFG

%PGM%\exec\hycs_std

REM -----------------------------------------------

REM For ConvLSTM we want no labels
echo ' ' >LABELS.CFG
%PGM%\exec\concplot -icdump -c50 -j%PGM%\graphics\arlmap -oconcplot_ml

REM Copy results to conc_output directory
copy concplot_ml.ps conc_output\
copy cdump conc_output\

REM Convert PS to image if needed for ConvLSTM (requires Ghostscript)
REM gswin64 -sDEVICE=png16m -r300 -o conc_output\frame%%d.png concplot_ml.ps

REM -----------------------------------------------
REM Create trajectory run
REM -----------------------------------------------

echo 25 01 08 15             >CONTROL
echo 1                      >>CONTROL
echo 34.03 -118.33 500      >>CONTROL
echo 24                     >>CONTROL
echo 0                      >>CONTROL
echo 15500                  >>CONTROL
echo 1                      >>CONTROL
echo C:/hysplit/HYSPLIT/    >>CONTROL
echo gdas1.jan25.w2         >>CONTROL
echo ./                     >>CONTROL
echo tdump                  >>CONTROL

IF EXIST tdump DEL tdump
IF EXIST SETUP.CFG DEL SETUP.CFG

%PGM%\exec\hyts_std

REM -----------------------------------------------
REM Create trajectory plot without labels
REM -----------------------------------------------

echo ' ' >LABELS.CFG
%PGM%\exec\trajplot -itdump -j%PGM%\graphics\arlmap -otrajplot_ml

REM Copy results to traj_output directory
copy trajplot_ml.ps traj_output\
copy tdump traj_output\

REM Convert PS to image if needed (requires Ghostscript)
REM gswin64 -sDEVICE=png16m -r300 -o traj_output\frame%%d.png trajplot_ml.ps

echo Finished! Output files are in conc_output and traj_output directories