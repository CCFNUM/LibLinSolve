// File       : coefficients.h
// Created    : Wed Apr 10 2024 13:27:38 (+0200)
// Author     : Fabian Wermelinger
// Description: Linear system coefficients
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef COEFFICIENTS_H_E818A4KY
#define COEFFICIENTS_H_E818A4KY

#include "CRSMatrix.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

namespace linearSolver
{

template <size_t N>
class coefficients : public CRSMatrix<N>
{
public:
    using Matrix = CRSMatrix<N>;
    using DataType = typename Matrix::DataType;
    using Vector = typename Matrix::Vector;
    using Index = typename Matrix::Index;
    using IndexVector = std::vector<Index>;
    using Matrix::BLOCKSIZE;

private:
    using Matrix::graph_;

public:
    struct LUData
    {
        IndexVector U_offsets; // offsets into U matrix (start from U_ii)
        Vector LU_values;      // LU factors
        Vector D_values;       // diagonal matrix
    };

    coefficients() = delete;

    coefficients(const CRSNodeGraph* graph, const Index id = 0)
        : CRSMatrix<N>(graph), id_(id)
    {
        resizeGraph();
        zeroALL();
    }

    KOKKOS_INLINE_FUNCTION
    Index getID() const
    {
        return id_;
    }

    void setGraph(const CRSNodeGraph* graph)
    {
        graph_ = *graph;
        resizeGraph();
    }

    void resizeGraph()
    {
        assert(graph_.isBuilt()); // MPI data is correct if graph is
                                  // built

        const Index n_local_coeff = graph_.nOwnedNodes();
        const Index n_local_ghosts = graph_.nGhostNodes();

        Kokkos::resize(this->values_, this->nnz()); // matrix coefficients
        Kokkos::resize(x_, BLOCKSIZE * (n_local_coeff + n_local_ghosts));
        Kokkos::resize(b_, BLOCKSIZE * n_local_coeff);
        Kokkos::resize(r_, BLOCKSIZE * n_local_coeff);
    }

    // clang-format off
    KOKKOS_INLINE_FUNCTION
    Matrix &getAMatrix() { return *static_cast<Matrix *>(this); }
    KOKKOS_INLINE_FUNCTION
    const Matrix &getAMatrix() const { return *static_cast<const Matrix *>(this); }
    KOKKOS_INLINE_FUNCTION
    Vector &getXVector() { return x_; }
    KOKKOS_INLINE_FUNCTION
    const Vector &getXVector() const { return x_; }
    KOKKOS_INLINE_FUNCTION
    Vector &getBVector() { return b_; }
    KOKKOS_INLINE_FUNCTION
    const Vector &getBVector() const { return b_; }
    KOKKOS_INLINE_FUNCTION
    Vector &getRVector() { return r_; }
    const Vector &getRVector() const { return r_; }

    // clang-format on

    void zeroALL()
    {
        zeroLHS();
        zeroRHS();
        zeroSOL();
        zeroRES();
    }

    void zeroLHS()
    {
        Kokkos::deep_copy(this->values_, 0);
    }

    void zeroRHS()
    {
        Kokkos::deep_copy(b_, 0);
    }

    void zeroSOL()
    {
        Kokkos::deep_copy(x_, 0);
    }

    void zeroRES()
    {
        Kokkos::deep_copy(r_, 0);
    }

    Index nCoefficients() const
    {
        return graph_.nOwnedNodes();
    }

    Index nGhostCoefficients() const
    {
        return graph_.nGhostNodes();
    }

    Index nGlobalCoefficients() const
    {
        return graph_.nGlobalRows();
    }

    LUData& getLUData()
    {
        return LU_data_;
    }

    const LUData& getLUData() const
    {
        return LU_data_;
    }

    void getMemoryFootprint(MemoryFootprint& data,
                            MemoryFootprint& connectivity) const
    {
        std::memset(&data, 0, sizeof(MemoryFootprint));

        const MPI_Comm comm = this->getCommunicator();

        // floating point data
        unsigned long long int n_elements = this->values_.size();
        n_elements += x_.size();
        n_elements += b_.size();
        n_elements += r_.size();
        n_elements += LU_data_.LU_values.size();
        n_elements += LU_data_.D_values.size();
        // clang-format off
        MPI_Reduce(&n_elements, &data.sum_byte, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
        // clang-format on
        using TReal = typename Matrix::DataType;
        data.sum_byte *= sizeof(TReal);

        // integer data
        this->graph_.getMemoryFootprint(connectivity);
    }

    void serialize(const std::string basename = "coeffs") const
    {
        const int rank = this->commRank();
        const int size = this->commSize();
        std::ostringstream fname;
        fname << basename << '.' << rank << '.' << size;
        std::ofstream fout(fname.str(), std::ios::binary);

        // graph
        this->graph_.serialize(fout);

        // coefficients (always write 64-bit)
        uint64_t v64;
        double fp64;

        // get an updated host mirror of data
        const auto lhs = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), this->values_);
        const auto rhs =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->b_);

        // header
        const char* p64 = reinterpret_cast<char*>(&v64);
        v64 = BLOCKSIZE;
        fout.write(p64, sizeof(uint64_t));
        v64 = lhs.size();
        fout.write(p64, sizeof(uint64_t));
        v64 = rhs.size();
        fout.write(p64, sizeof(uint64_t));

        // data
        p64 = reinterpret_cast<char*>(&fp64);
        for (size_t i = 0; i < lhs.extent(0); ++i)
        {
            fp64 = lhs(i);
            fout.write(p64, sizeof(double));
        }
        for (size_t i = 0; i < rhs.extent(0); ++i)
        {
            fp64 = rhs(i);
            fout.write(p64, sizeof(double));
        }
    }

    // Normalize matrix rows based on the row p1 norm
    void normalize()
    {
        using TReal = typename Matrix::DataType;

        auto& A = getAMatrix();
        auto& b = getBVector();

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

    void diagonalScale()
    {
        using TReal = typename Matrix::DataType;

        auto& A = getAMatrix();
        auto& b = getBVector();

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
            TReal scale[BLOCKSIZE] = {0};
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

private:
    const Index id_; // coefficient container ID

    Vector x_; // unknowns
    Vector b_; // right-hand-side
    Vector r_; // residual vector r = b - Ax

    // auxiliary data (not initialized nor resized during coefficient
    // construction since only required for specific solvers)
    LUData LU_data_;
};

} /* namespace linearSolver */

#endif /* COEFFICIENTS_H_E818A4KY */
