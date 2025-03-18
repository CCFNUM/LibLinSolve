# File       : CMakeLists.txt
# Created    : Sat Mar 15 2025 17:02:44 (+0100)
# Author     : Fabian Wermelinger
# Description: CMake build definitions
# Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.
cmake_minimum_required(VERSION 3.10)
project(LinearSolverLibrary VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

add_compile_options(
    -Wall
    -Wextra
    -pedantic
)

find_package(MPI REQUIRED)

set(libLinSolveIncludes
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/HYPRE"
    "${CMAKE_CURRENT_LIST_DIR}/PETSc"
    "${CMAKE_CURRENT_LIST_DIR}/matrix"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/test"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/layout"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/operators/blockMatrix"
    "${CMAKE_CURRENT_LIST_DIR}/matrix/operators/distributed"
    "${CMAKE_CURRENT_LIST_DIR}/utils"
)

set(libLinSolve "LinSolve")
add_library(${libLinSolve} SHARED
    "matrix/CRSNodeGraph.cpp"
    "utils/Profiler.cpp"
)
target_include_directories(${libLinSolve}
    PRIVATE
    ${MPI_INCLUDE_PATH}
    ${libLinSolveIncludes}
)

install(TARGETS ${libLinSolve} DESTINATION lib64)
install(FILES
  "AMG.h"
  "HYPRE/ContextHYPRE.h"
  "HYPRE/HYPRESolver.h"
  "PETSc/ContextPETSc.h"
  "PETSc/PETScSolver.h"
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

if(DEFINED WITH_FASTMATH AND WITH_FASTMATH)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -Ofast -mavx2 -mfma")
    string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -Ofast -mavx2 -mfma")
    message("-- Using fast math optimizations (NOT compliant with IEEE)")
endif()

get_filename_component(libAMG_PATH "libAMG" REALPATH)
if(EXISTS "${libAMG_PATH}")
    add_subdirectory(${libAMG_PATH})
    set(libAMG ${libAMG} PARENT_SCOPE)
endif()

set(libLinSolve ${libLinSolve} PARENT_SCOPE)
set(libLinSolve_INC ${libLinSolveIncludes} PARENT_SCOPE)
