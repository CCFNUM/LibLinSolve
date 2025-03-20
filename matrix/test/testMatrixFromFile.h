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
        : testSquareMatrix<N>(comm, layout, 10, conf)
    {
        assert(this->isLocalColumnOrder());

        if (conf)
        {
            if ((*conf)["filename"])
            {
                fname_ = (*conf)["filename"].template as<std::string>();
            }
            else
            {
                fname_ = "test_matrix.bin";
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

        std::copy(lhs_.begin(), lhs_.end(), a.begin());
        std::copy(rhs_.begin(), rhs_.end(), b.begin());
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
        std::ifstream fin(fname_, std::ios::binary);

        int blocksize;
        int n_rows;
        int nnz;
        fin.read((char*)&blocksize, sizeof(int));
        fin.read((char*)&n_rows, sizeof(int));
        fin.read((char*)&nnz, sizeof(int));

        this->n_ = n_rows;
        this->domainDecomposition(1);

        assert(blocksize == BLOCKSIZE);

        const Index n = this->getDimension();
        this->n_owned_nodes_ = n;
        row_ptr_.reserve(n + 1);
        primary_indices_.reserve(nnz);

        int v;
        for (int i = 0; i < n + 1; i++)
        {
            fin.read((char*)&v, sizeof(int));
            row_ptr_.push_back(v);
        }

        for (int i = 0; i < nnz; i++)
        {
            fin.read((char*)&v, sizeof(int));
            primary_indices_.push_back(v);
        }

        lhs_.reserve(blocksize * blocksize * nnz);
        rhs_.reserve(blocksize * n_rows);

        double src[BLOCKSIZE * BLOCKSIZE];
        for (int i = 0; i < nnz; i++)
        {
            fin.read((char*)src, blocksize * blocksize * sizeof(double));
            for (int j = 0; j < BLOCKSIZE * BLOCKSIZE; j++)
            {
                lhs_.push_back(static_cast<DataType>(src[j]));
            }
        }
        for (int i = 0; i < n; i++)
        {
            fin.read((char*)src, blocksize * sizeof(double));
            for (int j = 0; j < BLOCKSIZE; j++)
            {
                rhs_.push_back(static_cast<DataType>(src[j]));
            }
        }

        fin.close();
    }

    void computePackInfos_() override
    {
        // no communication required -> single rank only
        this->pack_infos_.clear();
    }
};

} /* namespace linearSolver */

#endif /* TESTMATRIXFROMFILE_H_VBZ1VOEM */
