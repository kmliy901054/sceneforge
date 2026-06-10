#!/usr/bin/env bash
# ROS Jazzy on this machine leaks a broken pytest plugin; keep plugins off and PYTHONPATH clean.
cd "$(dirname "$0")/.."
PY=${SCENEFORGE_PYTHON:-/home/pairlab/miniconda3/envs/dgan/bin/python}
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest tests/ -q "$@"
