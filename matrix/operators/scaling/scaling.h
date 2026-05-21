// File       : scaling.h
// Created    : Thu May 21 2026
// Author     : Mhamad Mahdi Alloush
// Description: Scaling operations for block systems
#ifndef SCALING_H_C0MCESRP
#define SCALING_H_C0MCESRP

#include "CRSMatrix.h"
#include <cmath>
#include <stdexcept>

namespace linearSolver
{

template <size_t N>
void normalize(CRSMatrix<N>& A,
               typename CRSMatrix<N>::Vector& b)
{
    using TReal = typename CRSMatrix<N>::DataType;
    using Index = typename CRSMatrix<N>::Index;
    static constexpr Index BLOCKSIZE = N;

    for (Index i = 0; i < A.nRows(); i++)
    {
        const auto block_col_idx = A.rowCols(i);
        const auto flat_block_row_values = A.rowVals(i);
        const Index n_col_blocks = block_col_idx.size();
        for (Index k = 0; k < BLOCKSIZE; k++)
        {
            // find p1-norm for the kth dof of the row
            TReal norm = 0.0;
            for (Index j = 0; j < n_col_blocks; j++)
            {
                for (Index l = 0; l < BLOCKSIZE; l++)
                {
                    const TReal& val =
                        flat_block_row_values[j * BLOCKSIZE * BLOCKSIZE +
                                              k * BLOCKSIZE + l];
                    norm += std::abs(val);
                }
            }

                   if (norm < 1e-30)
            {
                throw std::runtime_error(
                    "coefficients: Zero row norm in normalize()");
            }

            // normalize the kth dof of the row
            for (Index j = 0; j < n_col_blocks; j++)
            {
                for (Index l = 0; l < BLOCKSIZE; l++)
                {
                    TReal& val =
                        flat_block_row_values[j * BLOCKSIZE * BLOCKSIZE +
                                              k * BLOCKSIZE + l];
                    val /= norm;
                }
            }

                   // do for the rhs
            b[i * BLOCKSIZE + k] /= norm;
        }
    }
}

template <size_t N>
void diagonalScale(CRSMatrix<N>& A,
                   typename CRSMatrix<N>::Vector& b)
{
    using TReal = typename CRSMatrix<N>::DataType;
    using Index = typename CRSMatrix<N>::Index;
    static constexpr Index BLOCKSIZE = N;

    // Reused across rows; every entry is overwritten each iteration
    // (step 4) before it is read (step 5), so no per-row re-init needed.
    TReal scale[BLOCKSIZE];

    for (Index i = 0; i < A.nRows(); i++)
    {
        // 1) Get row structure (only reuse pointers)
        const auto block_col_idx = A.rowCols(i);
        auto flat_row_vals = A.rowVals(i);

        const Index* block_col_idx_ptr = block_col_idx.data();
        const Index n_col_blocks = static_cast<Index>(block_col_idx.size());
        TReal* flat_row_vals_ptr = flat_row_vals.data();

        // 2) Locate diagonal block A_ii
        Index diag_block_local_index = -1;
        for (Index j = 0; j < n_col_blocks; j++)
        {
            if (block_col_idx_ptr[j] == i)
            {
                diag_block_local_index = j;
                break;
            }
        }

        if (diag_block_local_index < 0)
        {
            throw std::runtime_error(
                "coefficients: Diagonal block missing in diagonalScale()");
        }

        // 3) Pointer to diagonal block
        const TReal* diag_block_ptr =
            flat_row_vals_ptr +
            diag_block_local_index * BLOCKSIZE * BLOCKSIZE;

        // 4) Compute scale[k] = 1 / A_ii(k,k)
        for (Index k = 0; k < BLOCKSIZE; k++)
        {
            const TReal diag_val = diag_block_ptr[k * BLOCKSIZE + k];

            if (std::abs(diag_val) < 1e-30)
            {
                throw std::runtime_error(
                    "coefficients: Zero diagonal entry in diagonalScale()");
            }

            scale[k] = static_cast<TReal>(1.0) / diag_val;
        }

        // 5) Scale each block in row i
        for (Index j = 0; j < n_col_blocks; j++)
        {
            TReal* block_ptr =
                flat_row_vals_ptr + j * BLOCKSIZE * BLOCKSIZE;

            for (Index k = 0; k < BLOCKSIZE; k++)
            {
                const TReal s = scale[k];

                for (Index l = 0; l < BLOCKSIZE; l++)
                {
                    block_ptr[k * BLOCKSIZE + l] *= s;
                }
            }
        }

        // 6) Scale the RHS block
        TReal* bi_ptr = &b[i * BLOCKSIZE];
        for (Index k = 0; k < BLOCKSIZE; k++)
        {
            bi_ptr[k] *= scale[k];
        }
    }
}

} /* namespace linearSolver */

#endif /* SCALING_H_C0MCESRP */