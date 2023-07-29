#!/usr/bin/env bash
set -e

USE_VENV=`pdm config python.use_venv`
VENV_DIR=`pdm config venv.location`

if ! command -v pdm &>/dev/null; then
    if ! command -v pipx &>/dev/null; then
        python3 -m pip install --user pipx
    fi
    pipx install pdm
fi
if ! pdm self list 2>/dev/null | grep -q pdm-multirun; then
    pdm install --plugins
fi

if [ -n "${PDM_MULTIRUN_VERSIONS}" ]; then
    if [ "${USE_VENV}" = "True" ]; then
        echo "Using Virtual Environments for python versions: ${PDM_MULTIRUN_VERSIONS}. Creating venvs"
        pdm multirun -fei ${PDM_MULTIRUN_VERSIONS// /,} -v pdm venv create --force
        pdm multirun -fei ${PDM_MULTIRUN_VERSIONS// /,} -v pdm install -G:all
    else
        pdm multirun -v pdm install -G:all
    fi
else
    pdm install -G:all
fi
