# File       : meson.build
# Created    : Fri Mar 14 2025 15:12:38 (+0100)
# Author     : Fabian Wermelinger
# Description: Build definitions
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

project('LinearSolverLibrary', ['cpp', 'c'],
  version: run_command(
    'git', 'tag', '--list', '--sort=-v:refname', 'v*', '--merged',
    capture: true, check: true).stdout().splitlines()[0].strip('v').strip('-'),
  license: 'BSD-2-Clause',
  meson_version: '>=1.2.0',
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
deps += dependency('dl', required: true)

# optional dependencies
petsc = dependency('PETSc', required: false)
hypre = dependency('HYPRE', required: false)
if petsc.found() and get_option('WITH_PETSC')
  deps += petsc
  if get_option('CONFIG')
    conf.set('HAS_PETSC', true)
  else
    cpp_args += '-DHAS_PETSC'
  endif
endif
if hypre.found() and get_option('WITH_HYPRE')
  deps += hypre
  if get_option('CONFIG')
    conf.set('HAS_HYPRE', true)
  else
    cpp_args += '-DHAS_HYPRE'
  endif
endif
if get_option('WITH_TRILINOS')
  deps += dependency('Trilinos', method: 'cmake', modules: ['Trilinos::all_libs'], required: true)
  if get_option('CONFIG')
    conf.set('HAS_TRILINOS', true)
  else
    cpp_args += '-DHAS_TRILINOS'
  endif
endif
if get_option('WITH_KOKKOS')
  deps += dependency('Kokkos', method: 'cmake', modules: ['Kokkos::all_libs'], required: true)
  if get_option('CONFIG')
    conf.set('HAS_KOKKOS', true)
  else
    cpp_args += '-DHAS_KOKKOS'
  endif
endif

src = [
  'AMGFactory.cpp',
  'matrix/CRSNodeGraph.cpp',
  'utils/Profiler.cpp',
]

hdr = include_directories(
  '.',
  'HYPRE',
  'PETSc',
  'Trilinos',
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
  build_rpath: meson.project_build_root() / 'libamg',
  cpp_args: cpp_args
)
libs = [libLinSolve]

fs = import('fs')
pkg_mod = import('pkgconfig')

has_amg = false
if get_option('WITH_AMG') and fs.is_dir('libamg')
  subdir('libamg')
  has_amg = true
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

# applications
subdir('apps/solver')

install_headers(
  'AMGFactory.h',
  'HYPRE/ContextHYPRE.h',
  'HYPRE/HYPRESolver.h',
  'PETSc/ContextPETSc.h',
  'PETSc/PETScSolver.h',
  'Trilinos/ContextTrilinos.h',
  'Trilinos/TrilinosSolver.h',
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

libLinSolve_dep = declare_dependency(
  include_directories: hdr,
  link_with: libs,
  dependencies: deps
)

pkg_mod.generate(
  name: 'LinSolve',
  filebase: 'LinSolve',
  description: 'HSLU linear solver library',
  libraries: libs
)
