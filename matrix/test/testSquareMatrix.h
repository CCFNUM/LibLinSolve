// File       : testSquareMatrix.h
// Created    : Sun May 12 2024 14:00:07 (+0200)
// Author     : Fabian Wermelinger
// Description: Test matrix base class
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef TESTSQUAREMATRIX_H_63WWNSDY
#define TESTSQUAREMATRIX_H_63WWNSDY

#include "CRSNodeGraph.h"
#include "linearSolverContext.h"
#include <iostream>
#include <mpi.h>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace linearSolver
{

template <size_t N>
class testSquareMatrix : public CRSNodeGraph
{
public:
    static constexpr size_t BLOCKSIZE = N;

    using Context = ::linearSolver::Context<N>;
    using Index = typename Context::Index;

    testSquareMatrix(const MPI_Comm comm,
                     const YAML::Node conf,
                     const Index n,
                     const GraphLayout layout)
        : CRSNodeGraph(comm, layout), n_(n)
    {
        if (conf)
        {
            if (conf["n"])
            {
                n_ = conf["n"].template as<Index>();
            }
        }

        domainDecomposition();
    }

    virtual void assemble(Context* ctx) = 0;

    virtual void report(Context* ctx)
    {
        assert(static_cast<const void*>(ctx->getCoefficients().getGraph()) ==
               static_cast<const void*>(this));

        const int rank = this->commRank();
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
        if (0 == rank)
        {
            std::cout << "Test square matrix domain row decomposition:\n";
            std::cout << "\tBlocksize: " << BLOCKSIZE << '\n';
            std::cout << "\tTotal number of rows: " << n_rows << '\n';
            for (int i = 0; i < size; i++)
            {
                std::cout << "\tRows on rank " << std::setw(3) << std::left << i
                          << ": " << decomposition[i] << '\n';
            }
        }
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
