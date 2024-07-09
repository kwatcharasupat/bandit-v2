#!/bin/bash

cleanup() {
    if type deactivate > /dev/null 2>&1; then
        deactivate
    fi
    rm -rf ./dependencies
    rm -rf ./pip_tools_cache
}

initiate() {
    python -m pip install virtualenv pip-tools
    mkdir -p ./pip_tools_cache

    python -m virtualenv dependencies
    source ./dependencies/bin/activate
}

cleanup
initiate

echo "----------------------------------------"
echo "Starting pip-compile for requirements.in"
echo "----------------------------------------"
echo
CUSTOM_COMPILE_COMMAND="./update_dependencies.sh" pip-compile requirements.in -v -o requirements.txt --upgrade --resolver=backtracking --cache-dir ./pip_tools_cache --pip-args="--no-cache-dir"

cleanup
