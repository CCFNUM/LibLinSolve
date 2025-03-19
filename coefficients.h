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
#include <memory>
#include <mpi.h>
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

    Index getID() const
    {
        return id_;
    }

    void setGraph(const CRSNodeGraph* graph)
    {
        graph_ = graph;
        resizeGraph();
    }

    void resizeGraph()
    {
        assert(graph_ && graph_->isBuilt()); // MPI data is correct if graph is
                                             // built

        const Index n_local_coeff = graph_->nOwnedNodes();
        const Index n_local_ghosts = graph_->nGhostNodes();

        this->values_.resize(this->nnz()); // matrix coefficients
        x_.resize(BLOCKSIZE * (n_local_coeff + n_local_ghosts));
        b_.resize(BLOCKSIZE * n_local_coeff);
        r_.resize(BLOCKSIZE * n_local_coeff);
    }

    // clang-format off
    Matrix &getAMatrix() { return *static_cast<Matrix *>(this); }
    const Matrix &getAMatrix() const { return *static_cast<const Matrix *>(this); }
    Vector &getXVector() { return x_; }
    const Vector &getXVector() const { return x_; }
    Vector &getBVector() { return b_; }
    const Vector &getBVector() const { return b_; }
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
        std::fill(this->values_.begin(), this->values_.end(), 0);
    }

    void zeroRHS()
    {
        std::fill(b_.begin(), b_.end(), 0);
    }

    void zeroSOL()
    {
        std::fill(x_.begin(), x_.end(), 0);
    }

    void zeroRES()
    {
        std::fill(r_.begin(), r_.end(), 0);
    }

    Index nCoefficients() const
    {
        return graph_->nOwnedNodes();
    }

    Index nGhostCoefficients() const
    {
        return graph_->nGhostNodes();
    }

    Index nGlobalCoefficients() const
    {
        return graph_->nGlobalRows();
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
        this->graph_->getMemoryFootprint(connectivity);
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
