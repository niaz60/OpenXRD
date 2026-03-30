#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
archive_path="${repo_root}/datasets/openxrd_dataset.zip"
output_dir="${repo_root}/data/openxrd"
acknowledge=0

usage() {
  cat <<'EOF'
Usage: ./scripts/unzip_dataset.sh --acknowledge [--output-dir PATH]

The OpenXRD dataset is evaluation-only.
It must not be used to train, fine-tune, distill, align, or otherwise optimize models.
Use of the dataset is conditioned on citing the accepted Digital Discovery paper.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --acknowledge)
      acknowledge=1
      shift
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "--output-dir requires a path." >&2
        exit 1
      fi
      output_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cat <<'EOF'
OpenXRD dataset policy:
  - OpenXRD is an evaluation-only dataset.
  - It must not be used to train, fine-tune, distill, align, or otherwise optimize models.
  - Use of the dataset is conditioned on citing the accepted Digital Discovery paper.

Why the dataset is zipped:
  The archive adds friction against casual scraping and accidental ingestion by
  online aggregation or crawling tools. It is not strong protection or access control.
EOF

if [[ "${acknowledge}" -ne 1 ]]; then
  echo >&2
  echo "Refusing to extract without explicit acknowledgment. Re-run with --acknowledge." >&2
  exit 2
fi

if [[ ! -f "${archive_path}" ]]; then
  echo "Dataset archive not found at ${archive_path}" >&2
  exit 1
fi

python_bin="$(command -v python3 || true)"
if [[ -z "${python_bin}" ]]; then
  python_bin="$(command -v python || true)"
fi
if [[ -z "${python_bin}" ]]; then
  echo "Python is required to extract the dataset archive." >&2
  exit 1
fi

"${python_bin}" - "${archive_path}" "${output_dir}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(output_dir)

print(f"Extracted {archive.name} to {output_dir}")
PY
