# Installs Inoculant for this user (%LOCALAPPDATA%\Programs\Inoculant, Start menu and Desktop shortcuts named "Atelier - Inoculant"), through install.bat.
& cmd /c "`"$PSScriptRoot\install.bat`" $args"
exit $LASTEXITCODE
