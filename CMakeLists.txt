# File       : CMakeLists.txt
# Created    : Sat Mar 15 2025 17:02:44 (+0100)
# Author     : Fabian Wermelinger
# Description: CMake build definitions (use meson build definitions for
#              standalone build)
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.
cmake_minimum_required(VERSION 3.12)
project(LinearSolverLibrary)
set(CMAKE_CXX_STANDARD 20)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

add_compile_options(
    -Wall
    -Wextra
    -pedantic
)


# the following region is special, do not remove the marker comments below
# %GIT_TAG%
# %EXTERNAL_REPLACE_START%
find_package(Git REQUIRED)
execute_process(
    COMMAND bash "-c" "git tag --list --sort=-v:refname v* --merged | head -n1"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# %EXTERNAL_REPLACE_END%
string(REGEX REPLACE "^v-?" "" CLEAN_VERSION "${GIT_DESCRIBE_VERSION}")
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" SEMVERSION "${CLEAN_VERSION}")
string(REGEX MATCH "^([0-9]+)" SOVERSION "${CLEAN_VERSION}")
if(SEMVERSION)
    message(STATUS "Found libLinSolve version: ${SEMVERSION}")
    set(PROJECT_VERSION_FULL "${CLEAN_VERSION}")
else()
    message(FATAL_ERROR "Cannot determine library version")
endif()

find_package(MPI REQUIRED)
# TODO: [fab4100@posteo.net; 2025-05-15] how to ask CMake which MPI
# implementation it has found?  Alternatively say explicitly which
# implementation to use like in Meson build definitions.
add_compile_options("-DOMPI_SKIP_MPICXX")
add_compile_options("-DMPICH_SKIP_MPICXX")

if(USE_KOKKOS)
    message(STATUS "Enabling Kokkos support")
    add_compile_options("-DUSE_KOKKOS")
    find_package(Kokkos REQUIRED)
endif()

# Optional Trilinos linear solver support
# Set -DTrilinos_DIR=<path> in CMake config
# Requires Belos, Ifpack2, and Tpetra packages to be enabled in Trilinos
if(WITH_TRILINOS_SOLVER AND NOT Trilinos_FOUND)
    find_package(Trilinos QUIET)
    if(Trilinos_FOUND)
        message(STATUS "Found Trilinos: ${Trilinos_DIR}")
        message(STATUS "Trilinos version: ${Trilinos_VERSION}")
        # Check if linear solver packages are available
        set(TRILINOS_LINEAR_SOLVER_AVAILABLE TRUE)
        foreach(pkg Belos Ifpack2 Tpetra)
            list(FIND Trilinos_PACKAGE_LIST ${pkg} pkg_index)
            if(pkg_index EQUAL -1)
                set(TRILINOS_LINEAR_SOLVER_AVAILABLE FALSE)
                message(STATUS "  Trilinos package ${pkg}: NOT FOUND")
            else()
                message(STATUS "  Trilinos package ${pkg}: found")
            endif()
        endforeach()
        if(TRILINOS_LINEAR_SOLVER_AVAILABLE)
            add_compile_definitions(HAS_TRILINOS)
            message(STATUS "Trilinos linear solver support: ENABLED")
        else()
            message(STATUS "Trilinos linear solver support: DISABLED (missing required packages)")
        endif()
    else()
        message(STATUS "Trilinos not found - Trilinos linear solver support disabled")
    endif()
endif()

set(libLinSolveIncludes
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/HYPRE"
    "${CMAKE_CURRENT_LIST_DIR}/PETSc"
    "${CMAKE_CURRENT_LIST_DIR}/Trilinos"
    "${CMAKE_CURRENT_LIST_DIR}/matrix"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/test"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/layout"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/operators/blockMatrix"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/operators/distributed"
    "${CMAKE_CURRENT_LIST_DIR}/utils"
)

if(NOT BUILD_SHARED_LIBS)
    add_compile_options("-DLIBLINSOLVE_STATIC")
endif()

set(libLinSolve "LinSolve")
add_library(${libLinSolve}
    "AMGFactory.cpp"
    "matrix/CRSNodeGraph.cpp"
    "utils/Profiler.cpp"
)
set_target_properties(${libLinSolve}
    PROPERTIES
    VERSION ${SEMVERSION}
    SOVERSION ${SOVERSION}
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/libamg"
)
target_include_directories(${libLinSolve}
    PRIVATE
    ${MPI_INCLUDE_PATH}
    ${libLinSolveIncludes}
)
target_link_libraries(${libLinSolve} PRIVATE ${MPI_CXX_LIBRARIES})

if(USE_KOKKOS)
    target_link_libraries(${libLinSolve} PRIVATE Kokkos::kokkos)
endif()

if(WITH_TRILINOS_SOLVER AND TRILINOS_LINEAR_SOLVER_AVAILABLE)
    target_include_directories(${libLinSolve}
        PRIVATE
        ${Trilinos_INCLUDE_DIRS}
        ${Trilinos_TPL_INCLUDE_DIRS}
    )
    target_link_libraries(${libLinSolve} PRIVATE
        ${Trilinos_LIBRARIES}
        ${Trilinos_TPL_LIBRARIES}
    )
endif()

install(TARGETS ${libLinSolve} DESTINATION lib64)
install(FILES
  "AMGFactory.h"
  "HYPRE/ContextHYPRE.h"
  "HYPRE/HYPRESolver.h"
  "PETSc/ContextPETSc.h"
  "PETSc/PETScSolver.h"
  "Trilinos/ContextTrilinos.h"
  "Trilinos/TrilinosSolver.h"
  "coefficients.h"
  "controlData.h"
  "linearSolver.h"
  "linearSolverContext.h"
  "linearSolverContext.hpp"
  "matrix/CRSConnectivity.h"
  "matrix/CRSMatrix.h"
  "matrix/CRSMatrix.hpp"
  "matrix/CRSNodeGraph.cpp"
  "matrix/CRSNodeGraph.h"
  "matrix/CRSNodeGraphSynchronizer.h"
  "matrix/layout/memoryLayout.h"
  "matrix/operators/blockMatrix/blockMatrixDeterminant.hpp"
  "matrix/operators/blockMatrix/blockMatrixInvert.hpp"
  "matrix/operators/blockMatrix/blockMatrixOperators.h"
  "matrix/operators/distributed/CRSMatrixVector.h"
  "matrix/operators/distributed/CRSResidual.h"
  "residual.h"
  "utils/Common.h"
  "utils/Profiler.h"
  DESTINATION include
)

if(WITH_FASTMATH)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -Ofast -mavx2 -mfma")
    string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -Ofast -mavx2 -mfma")
    message("-- Using fast math optimizations (NOT compliant with IEEE)")
endif()

if(WITH_AMGSOLVER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libamg")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libamg")

    # TODO [faw 2026-05-09]: add option for user specified path to libAMG.a to
    # support standalone AMG builds
    if(NOT BUILD_SHARED_LIBS)
        target_link_libraries(${libLinSolve} PRIVATE ${libAMG})
    endif()
endif()

set(libLinSolve ${libLinSolve} PARENT_SCOPE)
set(libLinSolve_INC ${libLinSolveIncludes} PARENT_SCOPE)
