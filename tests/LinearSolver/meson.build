# File       : meson.build
# Created    : Wed Mar 19 2025 16:58:58 (+0100)
# Author     : Fabian Wermelinger
# Description: Build rules for simple linear solver application
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

project('LinearSolver', ['cpp'],
  version: '0.1.0',
  license: 'BSD-2-Clause',
  default_options: [
    'cpp_std=c++20',
    'b_ndebug=if-release',
    'warning_level=3',
  ]
)

libLinSolve = dependency('LinSolve', required: true)
yaml_cpp = dependency('yaml-cpp', required: true)
deps = [libLinSolve, yaml_cpp]

mpi_implementation = get_option('MPI')
if mpi_implementation == 'mpich'
  deps += dependency('mpich', required: true)
elif mpi_implementation == 'openmpi'
  deps += dependency('ompi', required: true)
else
  error('Unknown MPI implementation', mpi_implementation)
endif
deps += dependency('PETSc', required: false)
deps += dependency('HYPRE', required: false)

executable('linear_solver_1',
  'LinearSolver.cpp',
  dependencies: deps,
  cpp_args: '-DBS=1'
)
executable('linear_solver_4',
  'LinearSolver.cpp',
  dependencies: deps,
  cpp_args: '-DBS=4'
)
