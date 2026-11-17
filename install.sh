#!/bin/sh
# Installs Inoculant for this user on Linux: ~/.local/share/inoculant and a menu entry. Run ./build.sh first.
set -e
cd "$(dirname "$0")"
SRC=build
[ -x ./Inoculant ] && SRC=.
DEST="$HOME/.local/share/inoculant"
mkdir -p "$DEST" "$HOME/.local/share/applications"
cp "$SRC/Inoculant" "$SRC/inoculant-cli" "$DEST/"
cp -r assets lang "$DEST/"
cp NOTICE LICENSE "$DEST/" 2>/dev/null || true
printf '[Desktop Entry]\nType=Application\nName=Atelier - Inoculant\nComment=Poisons and antidotes for your own pictures\nExec=%s/Inoculant\nIcon=%s/assets/logo.png\nCategories=Graphics;\n' "$DEST" "$DEST" > "$HOME/.local/share/applications/inoculant.desktop"
echo "Installed: $DEST"
