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
    'CONFIG=true',
  ]
)

# global compiler flags
cpp_args = []

conf = configuration_data()
if get_option('CONFIG')
  version = meson.project_version().split('.')
  conf.set('major', version[0])
  conf.set('minor', version[1])
  conf.set('patch', version[2])
  conf.set_quoted('version', meson.project_version())
else
  cpp_args += '-DNO_LINSOLVE_CONF'
endif

# dependencies
yaml_cpp = dependency('yaml-cpp', required: true)
deps = [yaml_cpp]

mpi_implementation = get_option('MPI')
if mpi_implementation == 'mpich'
  deps += dependency('mpich', required: true)
  cpp_args += '-DMPICH_SKIP_MPICXX'
elif mpi_implementation == 'openmpi'
  deps += dependency('ompi', required: true)
  cpp_args += '-DOMPI_SKIP_MPICXX'
else
  error('Unknown MPI implementation', mpi_implementation)
endif

# optional dependencies
petsc = dependency('PETSc', required: false)
hypre = dependency('HYPRE', required: false)
if petsc.found()
  deps += petsc
  if get_option('CONFIG')
    conf.set('HAS_PETSC', true)
  else
    cpp_args += '-DHAS_PETSC'
  endif
endif
if hypre.found()
  deps += hypre
  if get_option('CONFIG')
    conf.set('HAS_HYPRE', true)
  else
    cpp_args += '-DHAS_HYPRE'
  endif
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

if get_option('CONFIG')
  vcs = vcs_tag(
    command: [
      'git', 'describe', '--long', '--tags', '--dirty', '--broken', '--always'
      ],
    input: 'GitSHA.h.in',
    output: 'GitSHA.h'
  )
  src += vcs
endif

soversion = meson.project_version().split('.')[0]
libLinSolve = library('LinSolve',
  src,
  include_directories: hdr,
  dependencies: deps,
  install: true,
  soversion: soversion,
  version: meson.project_version(),
  cpp_args: cpp_args
)
libs = [libLinSolve]

fs = import('fs')
pkg_mod = import('pkgconfig')

if fs.is_dir('libamg')
  subdir('libamg')
endif

if get_option('CONFIG')
  configure_file(
    input: 'Config.h.in',
    output: 'Config.h',
    configuration: conf,
    install: true,
    install_dir: 'include'
  )
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
  install_dir: 'include'
)
install_subdir('matrix/test', install_dir : 'include')

pkg_mod.generate(
  name: 'LinSolve',
  filebase: 'LinSolve',
  description: 'HSLU linear solver library',
  libraries: libs
)
