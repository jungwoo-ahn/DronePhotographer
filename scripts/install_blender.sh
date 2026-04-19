#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BLENDER_DIR="${REPO_ROOT}/blender"
BLENDER_BIN="${BLENDER_DIR}/blender"

if [[ -x "${BLENDER_BIN}" ]]; then
    echo "Blender already installed: $("${BLENDER_BIN}" --version | head -1)"
    exit 0
fi

BLENDER_VERSION="${BLENDER_VERSION:-4.5.0}"
BLENDER_TAR="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="${BLENDER_URL:-https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/${BLENDER_TAR}}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

echo "Downloading Blender ${BLENDER_VERSION}..."
wget -q --show-progress -O "${TMPDIR}/${BLENDER_TAR}" "${BLENDER_URL}"

echo "Extracting..."
tar -xJf "${TMPDIR}/${BLENDER_TAR}" -C "${TMPDIR}"

EXTRACTED_DIR="$(ls -d "${TMPDIR}"/blender-* 2>/dev/null | head -1)"
if [[ -z "${EXTRACTED_DIR}" || ! -d "${EXTRACTED_DIR}" ]]; then
    echo "ERROR: Failed to find extracted Blender directory" >&2
    exit 1
fi

mv "${EXTRACTED_DIR}" "${BLENDER_DIR}"
echo "Installed: $("${BLENDER_BIN}" --version | head -1)"
