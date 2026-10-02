#!/usr/bin/env bash
#
# build.sh — Build the YOLO Inference desktop app and deploy it.
#
# What it does (in order):
#   1. Installs the JavaScript dependencies (npm install)
#   2. Builds the React frontend (vite build -> dist/)
#   3. Makes sure a Rust toolchain is available (installs a local one into
#      .toolchains/ if `cargo` is not already on the PATH)
#   4. Makes sure `nasm` is available (required the first time FFmpeg is being
#      compiled from source by ffmpeg-sys-next; a local copy is fetched into
#      .toolchains/bin if missing)
#   5. Runs `cargo build --release` inside src-tauri
#   6. Copies the resulting binary into deployment/
#   7. Syncs the ONNX models from src-tauri/models/ into deployment/models/
#
# Afterwards: `cd deployment && ./tauri_yolo_release` starts the app.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT="$ROOT/deployment"
TOOLCHAINS="$ROOT/.toolchains"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- npm --------
have npm   || die "npm is required but was not found on the PATH."
have node  || die "node is required but was not found on the PATH."

log "1/7 Installing JavaScript dependencies (npm install)"
npm --prefix "$ROOT" install --no-audit --no-fund

log "2/7 Building the React frontend (npm run build -> dist/)"
npm --prefix "$ROOT" run build

# -------------------------------------------------------------- rust ---------
if ! have cargo; then
  log "3/7 cargo not found — installing a local Rust toolchain into .toolchains/"
  mkdir -p "$TOOLCHAINS"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    RUSTUP_HOME="$TOOLCHAINS/rustup" CARGO_HOME="$TOOLCHAINS/cargo" \
    sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable
  export RUSTUP_HOME="${RUSTUP_HOME:-$TOOLCHAINS/rustup}"
  export CARGO_HOME="${CARGO_HOME:-$TOOLCHAINS/cargo}"
fi
# A local toolchain (if any) takes priority so the build is reproducible.
export PATH="$TOOLCHAINS/bin:$TOOLCHAINS/cargo/bin:$PATH"
have cargo || die "Rust (cargo) is required. Install it with: https://rustup.rs"
log "3/7 Using: $(cargo --version), $(rustc --version)"

# ---------------------------------------------------------------- nasm -------
if ! have nasm; then
  log "4/7 nasm not found — fetching nasm 2.16.03 from nasm.us into .toolchains/bin"
  mkdir -p "$TOOLCHAINS/bin"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -sSL -o "$tmp/nasm.rpm" \
    "https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/linux/nasm-2.16.03-0.fc39.x86_64.rpm" \
    || die "Could not download nasm from nasm.us."
  if have bsdtar; then
    bsdtar -xf "$tmp/nasm.rpm" -C "$tmp"
  elif have rpm2cpio; then
    (cd "$tmp" && rpm2cpio nasm.rpm | cpio -idm)
  else
    die "nasm is not installed and neither bsdtar nor rpm2cpio is available to extract the RPM. Install nasm via your package manager and re-run."
  fi
  cp -f "$tmp/usr/bin/nasm" "$TOOLCHAINS/bin/nasm"
fi
export PATH="$TOOLCHAINS/bin:$PATH"
have nasm || log "4/7 WARNING: no nasm found — the first FFmpeg build may fail (re-run once nasm is installed)."
log "4/7 Using: $(nasm --version 2>/dev/null || echo 'nasm (none)')"

# ------------------------------------------------------------- cargo ---------
# NOTE: the `tauri/custom-protocol` feature is REQUIRED for a production
# build. Without it the tauri crate compiles itself in "dev" mode and the
# app tries to load `devUrl` (Vite on localhost) instead of the bundled
# frontend, which shows "Could not connect to localhost: Connection refused".
log "5/7 Building the Tauri app (cargo build --release --features tauri/custom-protocol)"
(cd "$ROOT/src-tauri" && cargo build --release --features tauri/custom-protocol)

BIN="$ROOT/src-tauri/target/release/tauri_yolo_release"
[ -f "$BIN" ] || die "Release binary not found at $BIN"

# ------------------------------------------------------------- deploy --------
log "6/7 Copying binary into deployment/"
install -m 0755 "$BIN" "$DEPLOYMENT/tauri_yolo_release"

log "7/7 Syncing ONNX models into deployment/models/"
mkdir -p "$DEPLOYMENT/models"
shopt -s nullglob
for model in "$ROOT"/src-tauri/models/*.onnx; do
  cp -f "$model" "$DEPLOYMENT/models/"
  echo "    deployed: $(basename "$model")"
done
shopt -u nullglob

printf '\n\033[1;32mBuild + deployment finished.\033[0m\n'
echo "Start the app with:"
echo "    cd $DEPLOYMENT"
echo "    ./tauri_yolo_release"
