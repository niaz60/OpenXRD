#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
venv_dir="${repo_root}/.venv"

if command -v uv >/dev/null 2>&1; then
  uv venv --clear "${venv_dir}"
  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"
  uv pip install -e "${repo_root}"
else
  python_bin="${PYTHON_BIN:-}"
  if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python3.13 || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python3 || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    echo "Python was not found. Install Python 3.13 (recommended; 3.9+ supported) or uv first." >&2
    exit 1
  fi

  "${python_bin}" -m venv --clear "${venv_dir}"
  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -e "${repo_root}"
fi

cat <<'EOF'
OpenXRD is installed in .venv.

Next:
  source .venv/bin/activate
  ./scripts/unzip_dataset.sh --acknowledge
  openxrd-check
EOF
