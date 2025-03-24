@echo off
setlocal EnableDelayedExpansion

REM -----------------------------------------------
REM Set paths
REM -----------------------------------------------
set DIR=c:
set PGM=%DIR%\hysplit
set WORKDIR=%PGM%\working
cd %WORKDIR%

REM Create output directory for images if it doesn't exist
if not exist conc_images mkdir conc_images

echo Starting concentration plot generation...

REM -----------------------------------------------
REM Create empty LABELS.CFG (no labels for machine learning)
REM -----------------------------------------------
echo ' ' >LABELS.CFG

REM -----------------------------------------------
REM Generate concentration plots for each hour
REM -----------------------------------------------
set count=0
for /L %%h in (0,1,23) do (
    set hour=%%h
    if %%h LSS 10 set hour=0%%h
    
    echo Processing hour !hour! (%%h of 23)...
    
    REM Call concplot with display hour parameter (-d)
    %PGM%\exec\concplot -icdump -p1 -d%%h -j%PGM%\graphics\arlmap -oconcplot_h!hour!
    
    REM Convert PS to PNG using Ghostscript (if you have it installed)
    gswin64c -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r300 -sOutputFile=conc_images\hour!hour!.png concplot_h!hour!.ps
    
    REM Also copy PS files to the output directory
    copy concplot_h!hour!.ps conc_images\
    
    REM Clean up PS file in working directory
    del concplot_h!hour!.ps
    
    set /a count+=1
)

echo Finished! 
echo Concentration images are saved in: %WORKDIR%\conc_images
echo.
echo Generated !count! hourly images.
dir conc_images\*.png | find "file(s)"

pause