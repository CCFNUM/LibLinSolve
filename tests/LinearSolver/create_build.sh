#!/usr/bin/env sh
# File       : create_build.sh
# Created    : Wed Mar 19 2025 13:13:22 (+0100)
# Author     : Fabian Wermelinger
# Description: Create a build directory
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

if [ $# -lt 1 ]; then
    cat <<EOF
USAGE: $0 <path/to/build/dir> [mpich|openmpi] [meson setup args...]
EOF
    exit 1
fi

build_dir="${1}"; shift
if [ -d "${build_dir}" ]; then
    printf "Build directory '%s' already exists!\n" "${build_dir}"
    exit 1
fi

mpi_impl='mpich'
case ${1} in
    mpich) mpi_impl=mpich; shift;;
    openmpi) mpi_impl=openmpi; shift;;
esac

mkdir -p "${build_dir}"
cat <<EOF >"${build_dir}/build_env.sh"
module purge
module load gnu12
module load yaml-cpp
if [ "${mpi_impl}" = 'mpich' ]; then
    module load ucx
    module load mpich
    export LIBRARY_PATH=\${UCX_LIB}:\${LIBRARY_PATH}
elif [ "${mpi_impl}" = 'openmpi' ]; then
    module load openmpi4
else
    printf "Unknown MPI implementation '%s'!\n" "${mpi_impl}"
    exit 1
fi

module load petsc
export PKG_CONFIG_PATH=\${PETSC_LIB}/pkgconfig:\${PKG_CONFIG_PATH}

libHYPRE=\${LOCAL}/hypre_${mpi_impl}_release/lib64
export CMAKE_MODULE_PATH=\${libHYPRE}/cmake:\${CMAKE_MODULE_PATH}
export LD_LIBRARY_PATH=\${libHYPRE}:\${LD_LIBRARY_PATH}

export PKG_CONFIG_PATH=\${LOCAL}/linsolve_${mpi_impl}_release/lib64/pkgconfig:\${PKG_CONFIG_PATH}
EOF

source "${build_dir}/build_env.sh"
meson setup "${build_dir}" -DMPI=${mpi_impl} "${@}"
