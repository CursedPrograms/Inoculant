# Builds Inoculant and inoculant-cli (Release) and dist\Inoculant-win64.zip, through build.bat (Visual Studio 2022 + CMake).
& cmd /c "`"$PSScriptRoot\build.bat`""
exit $LASTEXITCODE
