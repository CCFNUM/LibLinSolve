#!/usr/bin/env sh
# File       : create_build.sh
# Created    : Wed Mar 19 2025 13:13:22 (+0100)
# Author     : Fabian Wermelinger
# Description: Create a build directory
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

if [ $# -lt 1 ]; then
    cat <<EOF
USAGE: $0 [--no-uenv] <path/to/build/dir> [meson setup args...]
EOF
    exit 1
fi

require_uenv=true
for arg; do
    case ${arg} in
        --no-uenv) require_uenv=false; shift ;;
    esac
done

build_dir="${1}"; shift
if [ -d "${build_dir}" ]; then
    printf "Build directory '%s' already exists!\n" "${build_dir}"
    exit 1
fi

if $require_uenv && [ -z "${UENV_MOUNT_LIST}" ] && [ -z "${UENV_VIEW}" ]; then
    echo "No uenv loaded..."
    exit 1
fi

mkdir -p "${build_dir}"

if $require_uenv; then
    cat <<EOF >"${build_dir}/build_env.sh"
export N_CORES=16
if ! grep -q '/user-environment' /proc/mounts; then
    unset UENV_VIEW UENV_ARG UENV_MOUNT_LIST UENV_TELEMETRY
    export BUILD_PREFIX='uenv run ${UENV_VIEW:+--view ${UENV_VIEW##*:}} ${UENV_LABEL:-$UENV_ARG} -- '
fi
EOF
fi

meson setup "${build_dir}" "${@}"
