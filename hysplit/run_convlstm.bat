@echo off
setLocal EnableDelayedExpansion

REM -----------------------------------------------
REM Set directories and create output folder
REM -----------------------------------------------

set DIR=c:
set PGM=%DIR%\hysplit
cd %PGM%\working

REM Create output directory if it doesn't exist
if not exist output mkdir output

REM -----------------------------------------------
REM Create ASCDATA.CFG
REM -----------------------------------------------

IF EXIST ASCDATA.CFG DEL ASCDATA.CFG
echo -90.0   -180.0  lat/lon of lower left corner   >ASCDATA.CFG
echo 1.0     1.0     lat/lon spacing in degrees    >>ASCDATA.CFG
echo 180     360     lat/lon number of data points >>ASCDATA.CFG
echo 2               default land use category     >>ASCDATA.CFG
echo 0.2             default roughness length (m)  >>ASCDATA.CFG
echo '%PGM%\bdyfiles\'  directory of files         >>ASCDATA.CFG

REM -----------------------------------------------
REM Run dispersion model (concentration) for each hour
REM -----------------------------------------------

REM Loop through 24 hours
for /L %%h in (1,1,24) do (
    set /a hour_duration=%%h
    
    echo 25 01 08 15             >CONTROL
    echo 1                      >>CONTROL
    echo 34.03 -118.33 500      >>CONTROL
    echo !hour_duration!        >>CONTROL
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
    echo 00 00 00 00 00         >>CONTROL
    echo 00 12 00               >>CONTROL
    echo 1                      >>CONTROL
    echo 0.0 0.0 0.0            >>CONTROL
    echo 0.0 0.0 0.0 0.0 0.0    >>CONTROL
    echo 0.0 0.0 0.0            >>CONTROL
    echo 0.0                    >>CONTROL
    echo 0.0                    >>CONTROL

    IF EXIST cdump DEL cdump
    IF EXIST SETUP.CFG DEL SETUP.CFG

    %PGM%\exec\hycs_std

    REM Create concentration plot for this hour
    ECHO 'TITLE^&','Hour %%h Concentration^&' >LABELS.CFG
    %PGM%\exec\concplot -icdump -c50 -j%PGM%\graphics\arlmap
    
    REM Convert PS to image using Ghostscript
    gswin64c -sDEVICE=png16m -o output\conc_hour_%%h.png -r300 concplot.ps
)

REM -----------------------------------------------
REM Run trajectory model for each hour
REM -----------------------------------------------

for /L %%h in (1,1,24) do (
    set /a hour_duration=%%h
    
    echo 25 01 08 15             >CONTROL
    echo 1                      >>CONTROL
    echo 34.03 -118.33 500      >>CONTROL
    echo !hour_duration!        >>CONTROL
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

    REM Create trajectory plot for this hour
    ECHO 'TITLE^&','Hour %%h Trajectory from Los Angeles^&' >LABELS.CFG
    %PGM%\exec\trajplot -itdump -j%PGM%\graphics\arlmap
    
    REM Convert PS to image using Ghostscript
    gswin64c -sDEVICE=png16m -o output\traj_hour_%%h.png -r300 trajplot.ps
)

REM -----------------------------------------------
REM Create GIFs from the sequences
REM -----------------------------------------------

REM For concentration images (using ImageMagick's convert command)
convert -delay 20 -loop 0 output\conc_hour_*.png output\concentration_animation.gif

REM For trajectory images
convert -delay 20 -loop 0 output\traj_hour_*.png output\trajectory_animation.gif

echo.
echo Process completed. Output files are in the 'output' directory.
echo concentration_animation.gif and trajectory_animation.gif have been created.
echo.