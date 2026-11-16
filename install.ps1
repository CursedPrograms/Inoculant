# Installs Inoculant for this user (%LOCALAPPDATA%\Programs\Inoculant, Start menu + Desktop shortcuts), through install.bat.
& cmd /c "`"$PSScriptRoot\install.bat`" $args"
exit $LASTEXITCODE
