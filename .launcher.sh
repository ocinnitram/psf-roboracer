# Sourced by ./train ./sweep ./select ./eval ./test ./rebuild ./track. Not executable on its own.
# cd to the repo root and make sure uv is present. The environment itself is
# pyproject.toml + uv.lock; `uv run` creates .venv on first use.
cd "$(dirname "${BASH_SOURCE[1]}")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
    echo "!! uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
# Clear PYTHONPATH so a sourced environment cannot shadow the project packages.
unset PYTHONPATH
