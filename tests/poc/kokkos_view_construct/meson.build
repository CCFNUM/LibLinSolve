# File       : meson.build
# Created    : Fri May 02 2025 17:36:16 (+0200)
# Author     : Fabian Wermelinger
# Description: Kokkos view construction proof of concept
# Copyright 2025 Fabian Wermelinger. All Rights Reserved.

project('KokkosViewConstructPOC', ['cpp'],
  version: '0.0.0',
  license: 'BSD-2-Clause',
  default_options: [
    'cpp_std=c++20',
    'b_ndebug=if-release',
    'warning_level=3',
  ]
)

# WIP: [fab4100@posteo.net; 2025-09-16] needs integration of nvcc_wrapper(?)
kokkos = dependency('Kokkos', method: 'cmake', modules: ['Kokkos::all_libs'], required: true)
openmp = dependency('openmp', required: true)
executable('kokkosViewConstructPOC',
  'main.cpp',
  dependencies: [kokkos, openmp],
)
