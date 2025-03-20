// File       : testSquareMatrix.h
// Created    : Sun May 12 2024 14:00:07 (+0200)
// Author     : Fabian Wermelinger
// Description: Test matrix base class
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef TESTSQUAREMATRIX_H_63WWNSDY
#define TESTSQUAREMATRIX_H_63WWNSDY

#include "CRSNodeGraph.h"
#include "linearSolverContext.h"
#include "residual.h"
#include <mpi.h>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace linearSolver
{

template <size_t N>
class testSquareMatrix : public CRSNodeGraph
{
public:
    using Context = ::linearSolver::Context<N>;
    using Index = typename Context::Index;

    static constexpr Index BLOCKSIZE = N;

    testSquareMatrix(const MPI_Comm comm,
                     const GraphLayout layout,
                     const Index n,
                     const YAML::Node* conf = nullptr)
        : CRSNodeGraph(comm, layout), n_(n)
    {
        if (conf)
        {
            if ((*conf)["n"])
            {
                n_ = (*conf)["n"].template as<Index>();
            }
        }

        domainDecomposition();
    }

    virtual void assemble(Context* ctx) = 0;

    virtual void report(Context* ctx)
    {
        assert(static_cast<const void*>(ctx->getCoefficients().getGraph()) ==
               static_cast<const void*>(this));

        const int size = this->commSize();

        Index n_rows = 0;
        std::vector<Index> decomposition(size);
        MPI_Reduce(&n_,
                   &n_rows,
                   1,
                   MPIDataType<Index>::type(),
                   MPI_SUM,
                   0,
                   this->comm_);
        MPI_Gather(&n_,
                   1,
                   MPIDataType<Index>::type(),
                   decomposition.data(),
                   1,
                   MPIDataType<Index>::type(),
                   0,
                   this->comm_);

        const auto ctrl = ctx->getControlData();
        auto& cout = ctx->cout();
        cout << "Test square matrix domain row decomposition:\n";
        cout << "\tBlocksize: " << BLOCKSIZE << '\n';
        cout << "\tTotal number of rows: " << n_rows << '\n';
        for (int i = 0; i < size; i++)
        {
            cout << "\tRows on rank " << std::setw(3) << std::left << i << ": "
                 << decomposition[i] << '\n';
        }
        cout << "\tInitial RMS:  " << ctrl.solver_initial_res << '\n';
        cout << "\tFinal RMS:    " << ctrl.solver_final_res << '\n';
        cout << "\tN iterations: " << ctrl.n_iterations << '\n';
    }

    operator CRSNodeGraph*() const
    {
        return this;
    }

    Index getDimension() const
    {
        return n_;
    }

protected:
    // matrix dimensions
    Index n_;

    void domainDecomposition(const int expect = 0)
    {
        // domain decomposition
        const int rank = this->commRank();
        const int size = this->commSize();
        if (expect > 0)
        {
            assert(expect == size);
        }

        Index my_n = this->n_ / size;
        const Index rem = this->n_ % size;
        if (rank >= size - rem)
        {
            ++my_n;
        }
        this->n_ = my_n;
    }
};

} /* namespace linearSolver */

#endif /* TESTSQUAREMATRIX_H_63WWNSDY */
