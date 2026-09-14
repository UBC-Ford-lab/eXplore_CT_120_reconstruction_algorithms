#!/usr/bin/env bash
# Build and install the X-ray Gaussian rasteriser that `--algorithm gaussian`
# needs, at the commit this package was measured against.
#
# Upstream (https://github.com/Ruyi-Zha/r2_gaussian) ships the extension with
# no version and no release tags, so a plain `pip install` of a fresh clone
# records nothing about the source it came from. This script
#   1. clones (or reuses) the upstream repository and checks out the pinned
#      commit — the same hash as `kernel.UPSTREAM_COMMIT`;
#   2. fetches the one nested submodule the build needs (glm; the sibling
#      simple-knn on gitlab.inria.fr is NOT needed and not fetched);
#   3. stamps the build's version as 0.0.0+g<commit>[.signed] so the pipeline
#      can report which build a run used (`kernel.build_info()`);
#   4. optionally applies the two-line signed-density patch (fabsf at the alpha
#      cutoff in forward.cu and backward.cu) that `--gauss-signed-density`
#      requires — the stock kernel never renders a negative contribution;
#   5. pip-installs it (non-editable) into the active Python, or builds a
#      wheel for a cluster node without a compiler.
# The build needs nvcc matching the installed torch (`--no-build-isolation`),
# exactly like TIGRE. Re-running is idempotent: the checkout is reset first.
set -euo pipefail

UPSTREAM_REPO="https://github.com/Ruyi-Zha/r2_gaussian"
UPSTREAM_COMMIT="f2579bfddd9aac009cb797c8503bef8119bbd022"   # keep equal to kernel.UPSTREAM_COMMIT
UPSTREAM_SUBDIR="r2_gaussian/submodules/xray-gaussian-rasterization-voxelization"
GLM_PATH="$UPSTREAM_SUBDIR/third_party/glm"

usage() {
    cat <<USAGE
usage: $(basename "$0") [--dir DIR] [--python PY] [--signed-density] [--wheel DIR] [--prepare-only] [--commit HASH]

  --dir DIR         where the upstream clone lives (default: \$XRAY_RASTERISER_SRC
                    or ~/third_party/r2_gaussian); created if missing
  --python PY       interpreter whose environment gets the build (default: the
                    active venv's python3)
  --signed-density  apply the signed-density patch; required by
                    --gauss-signed-density, harmless otherwise
  --wheel DIR       build a wheel into DIR instead of installing
  --prepare-only    clone / checkout / stamp / patch, but do not build
  --commit HASH     override the pinned commit (the pipeline will report the
                    build as NOT the pin; for re-measuring only)
USAGE
}

SRC_ROOT="${XRAY_RASTERISER_SRC:-$HOME/third_party/r2_gaussian}"
PYTHON="${PYTHON:-python3}"
SIGNED=0
WHEEL_DIR=""
PREPARE_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dir) SRC_ROOT="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --signed-density) SIGNED=1; shift ;;
        --wheel) WHEEL_DIR="$2"; shift 2 ;;
        --prepare-only) PREPARE_ONLY=1; shift ;;
        --commit) UPSTREAM_COMMIT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

say() { printf '==> %s\n' "$*"; }

# ---- 1. the checkout ------------------------------------------------------
if [ ! -d "$SRC_ROOT/.git" ]; then
    say "cloning $UPSTREAM_REPO into $SRC_ROOT"
    mkdir -p "$(dirname "$SRC_ROOT")"
    git clone --no-checkout "$UPSTREAM_REPO" "$SRC_ROOT"
fi
cd "$SRC_ROOT"
if ! git cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
    say "fetching $UPSTREAM_COMMIT"
    git fetch --quiet origin "$UPSTREAM_COMMIT" || git fetch --quiet origin
fi
say "checking out $UPSTREAM_COMMIT"
git -c advice.detachedHead=false checkout --quiet --force "$UPSTREAM_COMMIT"
git submodule update --init --quiet -- "$GLM_PATH"
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$UPSTREAM_COMMIT" ] || { echo "checkout is at $HEAD, not $UPSTREAM_COMMIT" >&2; exit 1; }
SHORT="${UPSTREAM_COMMIT:0:7}"
EXT="$SRC_ROOT/$UPSTREAM_SUBDIR"
[ -f "$EXT/setup.py" ] || { echo "no setup.py under $EXT — wrong commit or layout?" >&2; exit 1; }
[ -f "$EXT/third_party/glm/glm/glm.hpp" ] || { echo "glm submodule missing under $EXT/third_party/glm" >&2; exit 1; }

# ---- 2. reset, stamp, patch (idempotent: --force above restored the files) --
VERSION="0.0.0+g$SHORT"
FWD="$EXT/cuda_rasterizer/forward.cu"
BWD="$EXT/cuda_rasterizer/backward.cu"
if [ "$SIGNED" = 1 ]; then
    grep -q 'if (alpha < 0.00001f)' "$FWD" || { echo "forward.cu: alpha cutoff line not found; the patch does not apply to this commit" >&2; exit 1; }
    grep -q 'if (alpha <0.00001f)' "$BWD"  || { echo "backward.cu: alpha cutoff line not found; the patch does not apply to this commit" >&2; exit 1; }
    sed -i 's/if (alpha < 0.00001f)/if (fabsf(alpha) < 0.00001f)   \/\/ signed-density patch: a negative primitive must render and receive a gradient/' "$FWD"
    sed -i 's/if (alpha <0.00001f)/if (fabsf(alpha) < 0.00001f)   \/\/ signed-density patch/' "$BWD"
    VERSION="$VERSION.signed"
    say "signed-density patch applied"
fi
grep -q '^    name="xray_gaussian_rasterization_voxelization",$' "$EXT/setup.py" || { echo "setup.py layout changed; cannot stamp the version" >&2; exit 1; }
sed -i "s|^    name=\"xray_gaussian_rasterization_voxelization\",\$|    name=\"xray_gaussian_rasterization_voxelization\",\n    version=\"$VERSION\",|" "$EXT/setup.py"
say "source prepared at $EXT (version $VERSION)"
[ "$PREPARE_ONLY" = 1 ] && exit 0

# ---- 3. build --------------------------------------------------------------
"$PYTHON" -c 'import torch, torch.utils.cpp_extension' 2>/dev/null \
    || { echo "$PYTHON cannot import torch; activate the environment first (the build runs without isolation and compiles against the installed torch)" >&2; exit 1; }
if [ -n "$WHEEL_DIR" ]; then
    say "building a wheel into $WHEEL_DIR"
    "$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$WHEEL_DIR" "$EXT"
    ls -1 "$WHEEL_DIR"/xray_gaussian_rasterization_voxelization-*.whl
    exit 0
fi
say "installing into $("$PYTHON" -c 'import sys; print(sys.prefix)')"
"$PYTHON" -m pip install --no-build-isolation "$EXT"
"$PYTHON" - <<'PY'
from importlib import metadata
print("==> installed", metadata.version("xray_gaussian_rasterization_voxelization"))
PY
