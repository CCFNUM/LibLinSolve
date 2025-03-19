# File       : meson.build
# Created    : Fri Mar 14 2025 15:12:38 (+0100)
# Author     : Fabian Wermelinger
# Description: Build definitions
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

project('LinearSolverLibrary', ['cpp', 'c'],
  version: run_command(
    'git', 'tag', '--list', '--sort=-version:refname', 'v-*',
    capture: true, check: true).stdout().splitlines()[0].strip('v-'),
  license: 'BSD-2-Clause',
  default_options: [
    'cpp_std=c++20',
    'b_ndebug=if-release',
    'default_library=shared',
    'warning_level=3',
    'MPI=mpich',
  ]
)

# compiler object
cpp = meson.get_compiler('cpp')
c = meson.get_compiler('c')

# global compiler flags
add_global_arguments('-DOMPI_SKIP_MPICXX', language : 'cpp')

# dependencies
yaml_cpp = dependency('yaml-cpp', required: true)
deps = [yaml_cpp]

mpi_implementation = get_option('MPI')
if mpi_implementation == 'mpich'
  deps += dependency('mpich', required: true)
elif mpi_implementation == 'openmpi'
  deps += dependency('ompi', required: true)
else
  error('Unknown MPI implementation', mpi_implementation)
endif

# optional dependencies
petsc = dependency('PETSc', required: false)
hypre = dependency('HYPRE', required: false)
if petsc.found()
  add_global_arguments('-DHAS_PETSC', language : 'cpp')
  deps += petsc
endif
if hypre.found()
  add_global_arguments('-DHAS_HYPRE', language : 'cpp')
  deps += hypre
endif

src = [
  'matrix/CRSNodeGraph.cpp',
  'utils/Profiler.cpp',
]
hdr = include_directories(
  '.',
  'HYPRE',
  'PETSc',
  'matrix',
  'matrix/layout',
  'matrix/operators/blockMatrix',
  'matrix/operators/distributed',
  'utils',
)

soversion = meson.project_version().split('.')[0]
libLinSolve = library('LinSolve',
  src,
  include_directories: hdr,
  dependencies: deps,
  install: true,
  soversion: soversion,
  version: meson.project_version()
)
libs = [libLinSolve]

fs = import('fs')
pkg_mod = import('pkgconfig')

if fs.is_dir('libAMG')
  subdir('libAMG')
endif

install_headers(
  'AMG.h',
  'HYPRE/ContextHYPRE.h',
  'HYPRE/HYPRESolver.h',
  'PETSc/ContextPETSc.h',
  'PETSc/PETScSolver.h',
  'coefficients.h',
  'controlData.h',
  'linearSolver.h',
  'linearSolverContext.h',
  'linearSolverContext.hpp',
  'matrix/CRSConnectivity.h',
  'matrix/CRSMatrix.h',
  'matrix/CRSMatrix.hpp',
  'matrix/CRSNodeGraph.cpp',
  'matrix/CRSNodeGraph.h',
  'matrix/CRSNodeGraphSynchronizer.h',
  'matrix/layout/memoryLayout.h',
  'matrix/operators/blockMatrix/blockMatrixDeterminant.hpp',
  'matrix/operators/blockMatrix/blockMatrixInvert.hpp',
  'matrix/operators/blockMatrix/blockMatrixOperators.h',
  'matrix/operators/distributed/CRSMatrixVector.h',
  'matrix/operators/distributed/CRSResidual.h',
  'residual.h',
  'utils/Common.h',
  'utils/Profiler.h',
)

pkg_mod.generate(
  name: 'LinSolve',
  filebase: 'LinSolve',
  description: 'HSLU linear solver library',
  libraries: libLinSolve
)
