# File       : CMakeLists.txt
# Created    : Fri May 02 2025 22:34:49 (+0200)
# Author     : Fabian Wermelinger
# Description: Kokkos array strides proof of concept
# Copyright 2025 Fabian Wermelinger. All Rights Reserved.

cmake_minimum_required(VERSION 3.16)
project(KokkosArrayStridePOC)

find_package(Kokkos REQUIRED)
add_executable(kokkosArrayStridePOC main.cpp)
target_link_libraries(kokkosArrayStridePOC Kokkos::kokkos)
