#!/usr/bin/env sh
# File       : create_build.sh
# Created    : Wed Mar 19 2025 13:13:22 (+0100)
# Author     : Fabian Wermelinger
# Description: Create a build directory
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

if [ $# -lt 1 ]; then
    cat <<EOF
USAGE: $0 <path/to/build/dir> [meson setup args...]
EOF
    exit 1
fi

build_dir="${1}"; shift
if [ -d "${build_dir}" ]; then
    printf "Build directory '%s' already exists!\n" "${build_dir}"
    exit 1
fi

if [ -z "${UENV_ARG}" ]; then
    echo "UENV_ARG environment variable is empty (is a uenv loaded?)"
    exit 1
fi
mkdir -p "${build_dir}"
cat <<EOF >"${build_dir}/build_env.sh"
    export BUILD_PREFIX='uenv run ${UENV_VIEW:+--view ${UENV_VIEW##*:}} ${UENV_ARG} -- '
EOF
meson setup "${build_dir}" "${@}"
