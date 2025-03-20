// File       : testDiagonalMatrix.h
// Created    : Sun May 12 2024 14:30:05 (+0200)
// Author     : Fabian Wermelinger
// Description: Diagonal matrix
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef TESTDIAGONALMATRIX_H_T29S64DS
#define TESTDIAGONALMATRIX_H_T29S64DS

#include "blockMatrixOperators.h"
#include "testSquareMatrix.h"
#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <random>

namespace linearSolver
{

template <size_t N>
class testDiagonalMatrix : public testSquareMatrix<N>
{
public:
    using Base = testSquareMatrix<N>;
    using Base::BLOCKSIZE;
    using Base::diagonal_row_offset_;
    using Base::primary_indices_;
    using Base::row_ptr_;
    using Base::secondary_indices_;

    using Index = typename Base::Index;
    using Context = typename Base::Context;
    using Matrix = typename Context::Matrix;
    using Vector = typename Context::Vector;
    using DataType = typename Matrix::DataType;

    testDiagonalMatrix(const MPI_Comm comm,
                       const GraphLayout layout,
                       const YAML::Node* conf = nullptr)
        : testSquareMatrix<N>(comm, layout, 10, conf), b_i_(1.0), blk_ii_(2.0),
          blk_ij_(1.0), random_(false)
    {
        if (conf)
        {
            if ((*conf)["random"])
            {
                random_ = (*conf)["random"].template as<bool>();
            }
            if ((*conf)["b_i"])
            {
                b_i_ = (*conf)["b_i"].template as<DataType>();
            }
            if ((*conf)["blk_ii"])
            {
                blk_ii_ = (*conf)["blk_ii"].template as<DataType>();
            }
            if ((*conf)["blk_ij"])
            {
                blk_ij_ = (*conf)["blk_ij"].template as<DataType>();
            }
        }

        std::random_device random_seed;
        rng_.seed(random_seed());

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

        // initialize
        ctx->getCoefficients().zeroLHS();
        ctx->getCoefficients().zeroRHS();
        auto a = A.valuesRef();
        assert(static_cast<Index>(a.size()) == BLOCKSIZE * BLOCKSIZE * n);
        assert(static_cast<Index>(b.size()) == BLOCKSIZE * n);

        // set test system
        std::uniform_real_distribution<DataType> U(0, 1);
        for (Index i = 0; i < A.nRows(); i++)
        {
            DataType* b_i = &b[BLOCKSIZE * i];
            DataType* blk_i = &a[BLOCKSIZE * BLOCKSIZE * i];
            assert(blk_i == A.diag(i));
            setRHS_(b_i, &U);
            setLHS_(blk_i, &U);
        }
    }

    void report(Context* ctx) override
    {
        assert(static_cast<const void*>(ctx->getCoefficients().getGraph()) ==
               static_cast<const void*>(this));

        Base::report(ctx);

        const auto& x = ctx->getXVector();
        const Index n = this->getDimension();

        DataType b_i_exact[BLOCKSIZE];
        DataType blk_exact[BLOCKSIZE * BLOCKSIZE];
        setLHS_(blk_exact);
        setRHS_(b_i_exact);
        BlockMatrix::invertInplace<BLOCKSIZE>(blk_exact);
        BlockMatrix::matrixVectorInplace<BLOCKSIZE>(blk_exact, b_i_exact);

        double mse = 0;
        for (Index i = 0; i < n; i++)
        {
            const DataType* x_i = &x[BLOCKSIZE * i];
            for (Index k = 0; k < BLOCKSIZE; k++)
            {
                const double xk = x_i[k];
                const double e = xk - b_i_exact[k];
                mse += e * e;
            }
        }

        double gmse = 0;
        MPI_Reduce(&mse, &gmse, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_);

        auto& cout = ctx->cout();
        gmse /= ctx->getCoefficients().nGlobalCoefficients();
        cout << "Test diagonal matrix solution:\n";
        cout << "\tBase values:\n";
        cout << "\tb_i =    " << b_i_ << '\n';
        cout << "\tblk_ii = " << blk_ii_ << '\n';
        cout << "\tblk_ij = " << blk_ij_ << '\n';
        if (random_)
        {
            cout << "\tRANDOM COEFFICIENTS\n";
        }
        else
        {
            cout << "\tDETERMINISTIC COEFFICIENTS\n";
            cout << "\tEXPECTED = [";
            for (Index k = 0; k < BLOCKSIZE - 1; k++)
            {
                cout << std::scientific << b_i_exact[k] << ", ";
            }
            cout << std::scientific << b_i_exact[BLOCKSIZE - 1] << "]\n";
        }
        cout << "\tMSE = " << std::scientific << gmse << '\n';
    }

private:
    DataType b_i_;    // base RHS value
    DataType blk_ii_; // base block diagonal value
    DataType blk_ij_; // base block off-diagonal value

    bool random_;
    std::mt19937 rng_;

    void buildGraph_() override
    {
        const Index n = this->getDimension();

        this->n_owned_nodes_ = n;
        row_ptr_.resize(n + 1);
        row_ptr_[0] = 0;
        for (Index i = 0; i < n; i++)
        {
            row_ptr_[i + 1] = i + 1;
        }

        this->computeGlobalRowOffset_();
        Index column_shift = 0;
        if (this->isGlobalColumnOrder())
        {
            column_shift = this->global_row_offset_;
        }

        primary_indices_.resize(n);
        secondary_indices_.resize(n);
        assert(this->nIndices() == n); // nnz == n
        for (Index i_local = 0; i_local < this->nIndices(); i_local++)
        {
            const Index i_global = i_local + this->global_row_offset_;
            primary_indices_[i_local] = i_local + column_shift;
            secondary_indices_[i_local] = i_global - column_shift;
        }
    }

    void computePackInfos_() override
    {
        // no communication required for diagonal matrix
        this->pack_infos_.clear();
        this->n_ghost_nodes_ = 0;
    }

    void setLHS_(DataType* blk,
                 std::uniform_real_distribution<DataType>* U = nullptr)
    {
        if (U && random_)
        {
            for (Index m = 0; m < BLOCKSIZE; m++)
            {
                for (Index n = 0; n < BLOCKSIZE; n++)
                {
                    if (m == n)
                    {
                        blk[BLOCKSIZE * m + n] = blk_ii_ + U->operator()(rng_);
                    }
                    else
                    {
                        blk[BLOCKSIZE * m + n] = blk_ij_ + U->operator()(rng_);
                    }
                }
            }
        }
        else
        {
            for (Index m = 0; m < BLOCKSIZE; m++)
            {
                for (Index n = 0; n < BLOCKSIZE; n++)
                {
                    if (m == n)
                    {
                        blk[BLOCKSIZE * m + n] = blk_ii_;
                    }
                    else
                    {
                        blk[BLOCKSIZE * m + n] = blk_ij_;
                    }
                }
            }
        }
    }

    void setRHS_(DataType* b_i,
                 std::uniform_real_distribution<DataType>* U = nullptr)
    {
        if (U && random_)
        {
            for (Index i = 0; i < BLOCKSIZE; i++)
            {
                b_i[i] = b_i_ + U->operator()(rng_);
            }
        }
        else
        {
            for (Index i = 0; i < BLOCKSIZE; i++)
            {
                b_i[i] = b_i_;
            }
        }
    }
};

} /* namespace linearSolver */

#endif /* TESTDIAGONALMATRIX_H_T29S64DS */
