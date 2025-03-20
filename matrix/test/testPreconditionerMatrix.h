// File       : testPreconditionerMatrix.h
// Created    : Mon May 13 2024 12:38:47 (+0200)
// Author     : Fabian Wermelinger
// Description: Stiff matrix for use with convergence tests and preconditioners
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef TESTPRECONDITIONERMATRIX_H_ET6MOBJ0
#define TESTPRECONDITIONERMATRIX_H_ET6MOBJ0

#include "blockMatrixOperators.h"
#include "testSquareMatrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <random>
#include <span>
#include <unordered_set>

namespace linearSolver
{

template <size_t N>
class testPreconditionerMatrix : public testSquareMatrix<N>
{
public:
    using Base = testSquareMatrix<N>;
    using Base::BLOCKSIZE;
    using Base::primary_indices_;
    using Base::row_ptr_;
    using Base::secondary_indices_;

    using Index = typename Base::Index;
    using Context = typename Base::Context;
    using Matrix = typename Context::Matrix;
    using Vector = typename Context::Vector;
    using DataType = typename Matrix::DataType;

    testPreconditionerMatrix(const MPI_Comm comm,
                             const GraphLayout layout,
                             const YAML::Node* conf = nullptr)
        : testSquareMatrix<N>(comm, layout, 1000, conf), b_i_(1.0),
          blk_ii_(0.5), blk_ij_(1.0), off_diag_dominance_(1.0),
          off_diagonal_(100), random_(false)
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
            if ((*conf)["off_diag_dominance"])
            {
                off_diag_dominance_ =
                    (*conf)["off_diag_dominance"].template as<DataType>();
            }
            if ((*conf)["off_diagonal"])
            {
                off_diagonal_ = (*conf)["off_diagonal"].template as<Index>();
            }
        }

        const int rank = this->commRank();
        rng_.seed(rank);
        // std::random_device random_seed;
        // rng_.seed(random_seed());

        this->buildGraph();
    }

    void assemble(Context* ctx) override
    {
        Matrix& A = ctx->getAMatrix();
        Vector& b = ctx->getBVector();

        const Index n_local = this->getDimension();
        const Index offset = A.globalRowOffset();

        assert(static_cast<const void*>(A.getGraph()) ==
               static_cast<const void*>(this));
        assert(A.nRows() == n_local);

        // set test system
        ctx->getCoefficients().zeroLHS();
        ctx->getCoefficients().zeroRHS();

        Index shift = 0;
        if (this->isGlobalColumnOrder())
        {
            shift = offset;
        }
        std::uniform_real_distribution<DataType> U(0, 1);
        for (Index i_local = 0; i_local < n_local; i_local++)
        {
            const Index i = i_local + offset;
            const auto columns = A.rowCols(i_local);
            auto A_i = A.rowVals(i_local);
            for (Index j_idx = 0; j_idx < static_cast<Index>(columns.size());
                 j_idx++)
            {
                const Index j_local = columns[j_idx] - shift;
                const Index j = j_local + offset; // global

                DataType* b_i = &b[BLOCKSIZE * i_local];
                DataType* a_ij = &A_i[BLOCKSIZE * BLOCKSIZE * j_idx];

                setRHS_(b_i, &U);
                if (i == j)
                {
                    setLHSDiagonal_(a_ij, i, &U);
                }
                else
                {
                    setLHSOffDiagonal_(a_ij, &U);
                }
            }
        }
    }

    void report(Context* ctx) override
    {
        assert(static_cast<const void*>(ctx->getCoefficients().getGraph()) ==
               static_cast<const void*>(this));

        Base::report(ctx);

        auto& cout = ctx->cout();
        cout << "Test preconditioner matrix:\n";
        cout << "\tBase values:\n";
        cout << "\tb_i =                " << b_i_ << '\n';
        cout << "\tblk_ii =             " << blk_ii_ << '\n';
        cout << "\tblk_ij =             " << blk_ij_ << '\n';
        cout << "\toff_diag_dominance = " << off_diag_dominance_ << '\n';
        cout << "\toff_diagonal       = " << off_diagonal_ << '\n';
        if (random_)
        {
            cout << "\tRANDOM COEFFICIENTS\n";
        }
        else
        {
            cout << "\tDETERMINISTIC COEFFICIENTS\n";
        }
    }

private:
    DataType b_i_;                // base RHS value
    DataType blk_ii_;             // base block diagonal value
    DataType blk_ij_;             // base block off-diagonal value
    DataType off_diag_dominance_; // base block off-diagonal dominance scale
    Index off_diagonal_;

    bool random_;
    std::mt19937 rng_;

    void buildGraph_() override
    {
        using IndexVector = std::vector<Index>;

        Index n_global = 0;
        const Index n_local = this->getDimension();
        MPI_Allreduce(&n_local,
                      &n_global,
                      1,
                      MPIDataType<Index>::type(),
                      MPI_SUM,
                      this->comm_);

        this->n_owned_nodes_ = n_local;
        row_ptr_.resize(n_local + 1);
        row_ptr_[0] = 0;

        IndexVector row_local;
        IndexVector row_global;
        IndexVector* row_primary = &row_local;
        IndexVector* row_secondary = &row_global;
        if (this->isGlobalColumnOrder())
        {
            row_primary = &row_global;
            row_secondary = &row_local;
        }
        primary_indices_.clear();
        secondary_indices_.clear();

        Index nnz = 0;
        Index local_idx_ghost = n_local;
        IndexVector global_2_local(n_global, -1);
        this->computeGlobalRowOffset_();
        for (Index i = 0; i < n_local; i++)
        {
            const Index i_global = i + this->global_row_offset_;
            global_2_local[i_global] = i;
        }
        for (Index i_local = 0; i_local < n_local; i_local++)
        {
            row_local.clear();
            row_global.clear();
            const Index i_global = i_local + this->global_row_offset_;
            for (Index j_global = 0; j_global < n_global; j_global++)
            {
                if (
                    // sub-diagonal
                    j_global == i_global - 1 ||
                    // diagonal
                    j_global == i_global ||
                    // super-diagonal
                    j_global == i_global + 1 ||
                    // distant sub/super-diagonals
                    std::abs(i_global - j_global) % off_diagonal_ == 0)
                {
                    const Index j_local = j_global - this->global_row_offset_;
                    if (j_local < 0 || n_local <= j_local)
                    {
                        if (global_2_local[j_global] < 0)
                        {
                            global_2_local[j_global] = local_idx_ghost++;
                        }
                    }
                    row_local.push_back(global_2_local[j_global]);
                    row_global.push_back(j_global);
                    ++nnz;
                }
            }
            primary_indices_.insert(primary_indices_.end(),
                                    row_primary->begin(),
                                    row_primary->end());
            secondary_indices_.insert(secondary_indices_.end(),
                                      row_secondary->begin(),
                                      row_secondary->end());

            row_ptr_[i_local + 1] = nnz;
        }
        assert(static_cast<Index>(primary_indices_.size()) == nnz);
        assert(static_cast<Index>(secondary_indices_.size()) == nnz);

        this->sortPrimaryIndices_();
        this->determine_n_ghosts_ = true;
    }

    void setLHSDiagonal_(DataType* blk,
                         const Index i,
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
                        blk[BLOCKSIZE * m + n] =
                            blk_ii_ + U->operator()(rng_) +
                            std::sqrt(static_cast<DataType>(i + 1));
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
                        blk[BLOCKSIZE * m + n] =
                            blk_ii_ + std::sqrt(static_cast<DataType>(i + 1));
                    }
                    else
                    {
                        blk[BLOCKSIZE * m + n] = blk_ij_;
                    }
                }
            }
        }
    }

    void
    setLHSOffDiagonal_(DataType* blk,
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
                        blk[BLOCKSIZE * m + n] =
                            off_diag_dominance_ * blk_ij_ + U->operator()(rng_);
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
                        blk[BLOCKSIZE * m + n] = off_diag_dominance_ * blk_ij_;
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

#endif /* TESTPRECONDITIONERMATRIX_H_ET6MOBJ0 */
