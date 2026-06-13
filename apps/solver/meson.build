# File       : meson.build
# Created    : Wed Mar 19 2025 16:58:58 (+0100)
# Author     : Fabian Wermelinger
# Description: Build rules for simple linear solver application
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

libs = [libLinSolve]
if has_amg
  libs += libAMG
endif

fs.copyfile('problem.yaml')

executable('linear_solver_1',
  'LinearSolver.cpp',
  include_directories: hdr,
  dependencies: deps,
  link_with: libs,
  install: false,
  cpp_args: '-DBS=1' # scalar
)
executable('linear_solver_4',
  'LinearSolver.cpp',
  include_directories: hdr,
  dependencies: deps,
  link_with: libs,
  install: false,
  cpp_args: '-DBS=4' # blocksize=4
)
