@echo off
setlocal
cd /d "%~dp0"
rem Installs Inoculant for this user: %LOCALAPPDATA%\Programs\Inoculant, with Start menu and Desktop shortcuts named Atelier - Inoculant.
rem Run build.bat first (or unzip a release and run this beside it).
set "SRC=dist\Inoculant"
if exist "Inoculant.exe" set "SRC=."
if not exist "%SRC%\Inoculant.exe" ( echo Build it first: build.bat & exit /b 1 )
tasklist /fi "imagename eq Inoculant.exe" | find /i "Inoculant.exe" >nul
if not errorlevel 1 ( echo Inoculant is open: close it and run this again. & exit /b 1 )
set "INST=%LOCALAPPDATA%\Programs\Inoculant"
if not exist "%INST%" mkdir "%INST%"
xcopy /e /i /q /y "%SRC%\*" "%INST%\" >nul || ( echo Couldn't copy to %INST% & exit /b 1 )
powershell -NoProfile -Command "$s=New-Object -ComObject WScript.Shell; foreach($d in @([Environment]::GetFolderPath('Programs'),[Environment]::GetFolderPath('Desktop'))){ Remove-Item (Join-Path $d 'Inoculant.lnk') -Force -ErrorAction SilentlyContinue; $l=$s.CreateShortcut((Join-Path $d 'Atelier - Inoculant.lnk')); $l.TargetPath='%INST%\Inoculant.exe'; $l.WorkingDirectory='%INST%'; $l.IconLocation='%INST%\Inoculant.exe,0'; $l.Description='Inoculant: poisons and antidotes for your own pictures'; $l.Save() }"
echo Installed: %INST%
if /i not "%~1"=="--quiet" echo Start it from the Start menu or the Desktop.
exit /b 0
