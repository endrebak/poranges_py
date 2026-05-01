#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install-local.sh [--debug] [--skip-tests] [--local-ruranges-core | --ruranges-core PATH] [--current-python | --python PYTHON]

Build and install polaranges_py for local benchmarking.

Expected checkout layout:

  parent-dir/
    polaranges/
    polaranges_py/

polaranges_py/Cargo.toml depends on ../polaranges, so this script must be run
from the polaranges_py checkout or from one of its subdirectories.

Options:
  --debug                Build without --release.
  --skip-tests           Install and run the import smoke test, but skip pytest.
  --local-ruranges-core  Patch crates.io ruranges-core to ../ruranges-core for this build.
  --ruranges-core PATH   Patch crates.io ruranges-core to a specific local checkout.
  --current-python       Install into the current shell's python instead of .venv.
  --python PYTHON        Install into a specific benchmark Python interpreter.
  -h, --help             Show this help.
EOF
}

build_mode="--release"
run_tests=1
install_mode="venv"
requested_python=""
use_local_ruranges_core=0
requested_ruranges_core=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      build_mode=""
      ;;
    --skip-tests)
      run_tests=0
      ;;
    --local-ruranges-core)
      use_local_ruranges_core=1
      requested_ruranges_core=""
      ;;
    --ruranges-core)
      if [[ $# -lt 2 ]]; then
        echo "error: --ruranges-core requires a checkout path" >&2
        exit 2
      fi
      use_local_ruranges_core=1
      requested_ruranges_core="$2"
      shift
      ;;
    --current-python)
      install_mode="python"
      requested_python=""
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "error: --python requires a Python interpreter path" >&2
        exit 2
      fi
      install_mode="python"
      requested_python="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
polaranges_dir="$(cd "${repo_dir}/../polaranges" 2>/dev/null && pwd || true)"
ruranges_core_dir=""
venv_dir="${repo_dir}/.venv"

if [[ ! -f "${repo_dir}/Cargo.toml" || ! -f "${repo_dir}/pyproject.toml" ]]; then
  echo "error: could not find polaranges_py Cargo.toml and pyproject.toml in ${repo_dir}" >&2
  exit 1
fi

if [[ -z "${polaranges_dir}" || ! -f "${polaranges_dir}/Cargo.toml" ]]; then
  cat >&2 <<EOF
error: expected the private Rust polaranges checkout at:
  ${repo_dir}/../polaranges

Clone or check out the repositories as siblings:
  parent-dir/
    polaranges/
    polaranges_py/
EOF
  exit 1
fi

if ! grep -Eq '^name = "polaranges"$' "${polaranges_dir}/Cargo.toml"; then
  echo "error: ${polaranges_dir}/Cargo.toml is not the polaranges crate" >&2
  exit 1
fi

if [[ "${use_local_ruranges_core}" -eq 1 ]]; then
  if [[ -n "${requested_ruranges_core}" ]]; then
    ruranges_core_dir="$(cd "${requested_ruranges_core}" 2>/dev/null && pwd || true)"
  else
    ruranges_core_dir="$(cd "${repo_dir}/../ruranges-core" 2>/dev/null && pwd || true)"
  fi

  if [[ -z "${ruranges_core_dir}" || ! -f "${ruranges_core_dir}/Cargo.toml" ]]; then
    cat >&2 <<EOF
error: expected a local ruranges-core checkout at:
  ${requested_ruranges_core:-${repo_dir}/../ruranges-core}

Use --ruranges-core PATH to point at a different checkout.
EOF
    exit 1
  fi

  if ! grep -Eq '^name = "ruranges-core"$' "${ruranges_core_dir}/Cargo.toml"; then
    echo "error: ${ruranges_core_dir}/Cargo.toml is not the ruranges-core crate" >&2
    exit 1
  fi
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required. Install Rust from https://rustup.rs/." >&2
  exit 1
fi

if command -v python3.11 >/dev/null 2>&1; then
  bootstrap_python="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  bootstrap_python="python3"
else
  echo "error: Python 3.11 or newer is required." >&2
  exit 1
fi

"${bootstrap_python}" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("error: Python 3.11 or newer is required.")
PY

cd "${repo_dir}"

echo "Using polaranges_py: ${repo_dir}"
echo "Using polaranges:    ${polaranges_dir}"
if [[ "${use_local_ruranges_core}" -eq 1 ]]; then
  echo "Using ruranges-core: ${ruranges_core_dir}"
else
  echo "Using ruranges-core: crates.io"
fi
echo "Using cargo:       $(command -v cargo)"

if [[ "${install_mode}" == "venv" ]]; then
  echo "Bootstrap Python:  $(${bootstrap_python} --version)"

  if [[ ! -d "${venv_dir}" ]]; then
    echo "Creating virtual environment: ${venv_dir}"
    "${bootstrap_python}" -m venv "${venv_dir}"
  fi

  # shellcheck source=/dev/null
  source "${venv_dir}/bin/activate"
  install_python="python"
else
  if [[ -n "${requested_python}" ]]; then
    install_python="${requested_python}"
  elif command -v python >/dev/null 2>&1; then
    install_python="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    install_python="$(command -v python3)"
  else
    echo "error: could not find python. Pass --python /path/to/python." >&2
    exit 1
  fi

  if [[ ! -x "${install_python}" ]]; then
    echo "error: Python interpreter is not executable: ${install_python}" >&2
    exit 1
  fi
fi

"${install_python}" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("error: install Python must be 3.11 or newer.")
PY

echo "Install Python:    $("${install_python}" --version) at $("${install_python}" -c 'import sys; print(sys.executable)')"

"${install_python}" -m pip install --upgrade pip maturin
if [[ "${run_tests}" -eq 1 ]]; then
  "${install_python}" -m pip install --upgrade pytest
fi

maturin_args=(develop)
if [[ "${use_local_ruranges_core}" -eq 1 ]]; then
  maturin_args+=(--config "patch.crates-io.ruranges-core.path=\"${ruranges_core_dir}\"")
fi
if [[ -n "${build_mode}" ]]; then
  maturin_args+=("${build_mode}")
fi

echo "Building and installing polaranges_py with maturin ${maturin_args[*]}"
"${install_python}" -m maturin "${maturin_args[@]}"

"${install_python}" - <<'PY'
import polaranges

print(f"polaranges import OK: {polaranges.benchmark_version()}")
PY

if [[ "${run_tests}" -eq 1 ]]; then
  "${install_python}" -m pytest
fi

if [[ "${install_mode}" == "venv" ]]; then
  cat <<EOF

Local polaranges_py install is ready.

Activate it with:
  source ${venv_dir}/bin/activate
EOF
else
  cat <<EOF

Local polaranges_py install is ready in:
  $("${install_python}" -c 'import sys; print(sys.executable)')
EOF
fi
