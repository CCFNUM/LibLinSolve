// File       : testMatrixFromFile.h
// Created    : Tue May 21 2024 18:18:14 (+0200)
// Author     : Fabian Wermelinger
// Description: Test matrix loaded from file
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef TESTMATRIXFROMFILE_H_VBZ1VOEM
#define TESTMATRIXFROMFILE_H_VBZ1VOEM

#include "blockMatrixOperators.h"
#include "testSquareMatrix.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <mpi.h>
#include <sstream>
#include <string>

namespace linearSolver
{

template <size_t N>
class testMatrixFromFile : public testSquareMatrix<N>
{
public:
    using Base = testSquareMatrix<N>;
    using Base::BLOCKSIZE;
    using Base::primary_indices_;
    using Base::row_ptr_;

    using Index = typename Base::Index;
    using Context = typename Base::Context;
    using Matrix = typename Context::Matrix;
    using Vector = typename Context::Vector;
    using DataType = typename Matrix::DataType;

    testMatrixFromFile(const MPI_Comm comm,
                       const GraphLayout layout,
                       const YAML::Node* conf = nullptr)
        : testSquareMatrix<N>(comm, layout, 10, conf), fname_("test_matrix.bin")
    {
        if (conf)
        {
            if ((*conf)["filename"])
            {
                fname_ = (*conf)["filename"].template as<std::string>();
            }
        }

        this->buildGraph();
    }

    void assemble(Context* ctx) override
    {
        Matrix& A = ctx->getAMatrix();
        Vector& b = ctx->getBVector();

        const Index n = this->getDimension();

        assert(static_cast<const void*>(A.getGraph()) ==
               static_cast<const void*>(this));
        assert(A.nRows() == n);

        auto a = A.valuesRef();
        assert(a.size() == lhs_.size());
        assert(b.size() == rhs_.size());

        Kokkos::deep_copy(lhs_, a);
        Kokkos::deep_copy(rhs_, b);
    }

    void report(Context* ctx) override
    {
        assert(static_cast<const void*>(ctx->getCoefficients().getGraph()) ==
               static_cast<const void*>(this));
        Base::report(ctx);
    }

private:
    std::string fname_;
    Vector lhs_;
    Vector rhs_;

    void buildGraph_() override
    {
        int rank;
        int size;
        MPI_Comm_rank(this->getCommunicator(), &rank);
        MPI_Comm_size(this->getCommunicator(), &size);

        std::ostringstream fname;
        fname << fname_ << '.' << rank << '.' << size;

        std::ifstream fin(fname.str(), std::ios::binary);
        this->deserialize(fin);
        this->n_ = this->n_owned_nodes_;

        uint64_t blocksize;
        uint64_t lhs_size;
        uint64_t rhs_size;
        fin.read((char*)&blocksize, sizeof(uint64_t));
        fin.read((char*)&lhs_size, sizeof(uint64_t));
        fin.read((char*)&rhs_size, sizeof(uint64_t));
        assert(blocksize == BLOCKSIZE);

        Kokkos::resize(lhs_, lhs_size);
        Kokkos::deep_copy(lhs_, Index{0});
        Kokkos::resize(rhs_, lhs_size);
        Kokkos::deep_copy(rhs_, Index{0});

        double fp64;
        char* p64 = reinterpret_cast<char*>(&fp64);
        for (size_t i = 0; i < lhs_.size(); i++)
        {
            fin.read(p64, sizeof(double));
            lhs_[i] = static_cast<DataType>(fp64);
        }
        for (size_t i = 0; i < rhs_.size(); i++)
        {
            fin.read(p64, sizeof(double));
            rhs_[i] = static_cast<DataType>(fp64);
        }
        fin.close();

        this->determine_n_ghosts_ = true;
    }
};

} /* namespace linearSolver */

#endif /* TESTMATRIXFROMFILE_H_VBZ1VOEM */
