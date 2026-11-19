#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "run without sudo"; exit 1;
fi

export PATH="$HOME/.local/bin:$PATH"

step() {
  echo
  echo ">> $1"
}

step "Welcome to Jetson Orin Nano setup script for Mini-Spot."

install_apt() {
  if dpkg -s "$1" >/dev/null; then
    step "$1 already installed."
  else
    step "Installing $1..."
    sudo apt-get install -y "$1"
    step "$1 successfully installed."
  fi
}

step "system packages:"
sudo apt-get update
install_apt "git"
install_apt "curl"

step "serial access:"
if groups "$USER" | grep -qw dialout; then
  step "$USER is already added to the dialout group."
else
  step "adding $USER to dialout group."
  sudo usermod -aG dialout "$USER"
  step "!!!!Group related changes only takes effect after logging in again!!!!"
fi

if [ -x "$HOME/.platformio/penv/bin/platformio" ] && [ -L "$HOME/.local/bin/platformio" ]; then
  step "PlatformIO is already installed."
else
  step "installing PlatformIO..."
  curl -fsSL -o get-platformio.py https://raw.githubusercontent.com/platformio/platformio-core-installer/master/get-platformio.py
  python3 get-platformio.py
  mkdir -p "$HOME/.local/bin"
  ln -sf "$HOME/.platformio/penv/bin/platformio" "$HOME/.local/bin/platformio"
  ln -sf "$HOME/.platformio/penv/bin/pio" "$HOME/.local/bin/pio"
  ln -sf "$HOME/.platformio/penv/bin/piodebuggdb" "$HOME/.local/bin/piodebuggdb"
  step "PlatformIO successfully installed."
fi

build_uv() {
  if [ -f "$HOME/pyproject.toml" ]; then
    step "found the file."
    uv sync --extra cpu --project "$HOME"
  else
    step "pyproject.toml not found in $HOME."
  fi
}

if [ -x "$HOME/.local/bin/uv" ]; then
  step "uv is already installed."
  build_uv
else
  step "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  step "uv successfully installed."
  build_uv
fi

