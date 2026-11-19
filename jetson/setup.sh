#!/usr/bin/env bash
set -euo pipefail


step() {
  echo
  echo ">> $1"
}

step "Welcome to Jetson Orin Nano setup script for Mini-Spot."

install_apt() {
  if command -v "$1" >/dev/null; then
    echo " $1 already installed"
  else
    sudo apt-get install -y "$1"
  fi
}

step "system packages"
sudo apt-get update
install_apt "git"
install_apt "curl"

step "serial access"
if groups "$USER" | grep -qw dialout; then
  echo " $USER already added to the dialout group"
else
  sudo usermod -aG dialout "$USER"
  echo "Group related changes only takes effect after logging in again!!!!"
fi

