// File       : LinearSolver.cpp
// Created    : Wed Mar 19 2025 15:07:02 (+0100)
// Author     : Fabian Wermelinger
// Description: Linear solver application
// Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.

#include <AMG.h>
#include <HYPRESolver.h>
#include <PETScSolver.h>
#include <test/testDiagonalMatrix.h>
#include <test/testMatrixFromFile.h>
#include <test/testPreconditionerMatrix.h>
#include <test/testSquareMatrix.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <linearSolver.h>
#include <mpi.h>
#include <string>
#include <yaml-cpp/yaml.h>
#ifdef HAS_PETSC
#include <petscsys.h>
#if PETSC_VERSION_LT(3, 18, 1)
#define ErrorWrapPetscCall(c) CHKERRQ(c)
#else
#define ErrorWrapPetscCall(c) PetscCall(c)
#endif
#endif /* HAS_PETSC */
#ifdef HAS_HYPRE
#include <HYPRE_utilities.h>
#endif /* HAS_HYPRE */

template <size_t BLOCKSIZE>
::linearSolver::Base<BLOCKSIZE>* newSolver(const YAML::Node& config,
                                           ::linearSolver::GraphLayout& layout)
{
    using LinearSolver = ::linearSolver::Base<BLOCKSIZE>;
    using Context = typename LinearSolver::Context;

    const YAML::Node& solver_lookup = config["solver_lookup"];
    const YAML::Node& solver_conf =
        solver_lookup[config["solver"].template as<std::string>()];

    std::string family_type;
    if (solver_conf["family"])
    {
        family_type = solver_conf["family"].template as<std::string>();
    }
    else
    {
        std::cerr << "Solver `family` not specified in configuration!\n";
        std::exit(1);
    }
    Context::tolower(family_type);

    LinearSolver* solver = nullptr;
    if (family_type == "petsc")
    {
#ifdef HAS_PETSC
        solver = new ::linearSolver::PETSc<BLOCKSIZE>(solver_conf);
        layout = ::linearSolver::GraphLayout::ColumnIndexOrder__Global;
#endif /* HAS_PETSC */
    }
    else if (family_type == "hypre")
    {
#ifdef HAS_HYPRE
        solver = new ::linearSolver::HYPRE<BLOCKSIZE>(solver_conf);
        layout = ::linearSolver::GraphLayout::ColumnIndexOrder__Global;
#endif /* HAS_HYPRE */
    }
    else if (family_type == "amgsolver")
    {
#ifdef HAS_AMG
        solver =
            static_cast<LinearSolver*>(::linearSolver::getAMGSolverInstance(
                BLOCKSIZE, solver_conf, solver_lookup, layout));
#endif /* HAS_AMG */
    }
    else if (family_type == "gmres")
    {
#ifdef HAS_AMG
        solver =
            static_cast<LinearSolver*>(::linearSolver::getGMRESSolverInstance(
                BLOCKSIZE, solver_conf, solver_lookup, layout));
#endif /* HAS_AMG */
    }
    else if (family_type == "directsolver")
    {
#ifdef HAS_AMG
        solver =
            static_cast<LinearSolver*>(::linearSolver::getDirectSolverInstance(
                BLOCKSIZE, solver_conf, layout));
#endif /* HAS_AMG */
    }

    return solver;
}

template <size_t BLOCKSIZE>
::linearSolver::testSquareMatrix<BLOCKSIZE>*
newMatrix(const YAML::Node& config, ::linearSolver::GraphLayout& layout)
{
    const std::string type = config["type"].template as<std::string>();
    if (type == "diagonal")
    {
        return new ::linearSolver::testDiagonalMatrix<BLOCKSIZE>(
            MPI_COMM_WORLD, layout, &config);
    }
    else if (type == "fromfile")
    {
        return new ::linearSolver::testMatrixFromFile<BLOCKSIZE>(
            MPI_COMM_WORLD, layout, &config);
    }
    else if (type == "preconditioner")
    {
        return new ::linearSolver::testPreconditionerMatrix<BLOCKSIZE>(
            MPI_COMM_WORLD, layout, &config);
    }
    else
    {
        return nullptr;
    }
}

int main(int argc, char* argv[])
{
    static constexpr size_t BLOCKSIZE = BS;
    using LinearSolver = ::linearSolver::Base<BLOCKSIZE>;
    using Context = typename LinearSolver::Context;
    using Matrix = ::linearSolver::testSquareMatrix<BLOCKSIZE>;

    MPI_Init(&argc, &argv);
#ifdef HAS_PETSC
    // Initialize the Petsc environment
    ErrorWrapPetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
#endif /* HAS_PETSC */
#ifdef HAS_HYPRE
    HYPRE_Initialize();
#endif /* HAS_HYPRE */

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool isroot = (0 == rank);

    if (2 != argc && isroot)
    {
        std::cerr << "Usage: " << argv[0] << " <yaml_config_file>\n";
        std::exit(1);
    }
    const YAML::Node config = YAML::LoadFile(argv[1]);

    // create linear solver
    ::linearSolver::GraphLayout layout = ::linearSolver::GraphLayout::UNDEFINED;
    LinearSolver* solver = newSolver<BLOCKSIZE>(config, layout);
    assert(solver);

    // create matrix
    Matrix* matrix = newMatrix<BLOCKSIZE>(config["matrix"], layout);
    assert(matrix);
    Context* ctx = solver->createContext("TestMatrix", matrix).get();
    matrix->assemble(ctx);
    if (isroot)
    {
        std::cout << "Matrix norm: " << std::scientific
                  << ctx->getAMatrix().norm() << '\n';
    }

    std::cout << "BANDWIDTH: " << ctx->getAMatrix().bandwidth() << '\n';
    std::cout << "PROFILE:   " << ctx->getAMatrix().profile() << '\n';

    solver->solve();
    solver->report();
    matrix->report(ctx);

    delete matrix;
    delete solver;

#ifdef HAS_HYPRE
    HYPRE_Finalize();
#endif /* HAS_HYPRE */
#ifdef HAS_PETSC
    // Finalize the Petsc environment.
    ErrorWrapPetscCall(PetscFinalize());
#endif /* HAS_PETSC */
    MPI_Finalize();

    return 0;
}
