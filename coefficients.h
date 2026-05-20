// File       : coefficients.h
// Created    : Wed Apr 10 2024 13:27:38 (+0200)
// Author     : Fabian Wermelinger
// Description: Linear system coefficients
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef COEFFICIENTS_H_E818A4KY
#define COEFFICIENTS_H_E818A4KY

#include "CRSMatrix.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
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

    // Schur-complement augmented constrained system:
    //
    //     A*x + G*lambda = b
    //     H*x + M*lambda = bc
    //
    // with block form:
    //
    //     [ A  G ] [ x      ] = [ b  ]
    //     [ H  M ] [ lambda ]   [ bc ]
    //
    // where:
    //
    //   A, b   : original system matrix and right-hand-side, respectively.
    //            The original problem is not assumed to be singular or
    //            underdetermined; the augmented system represents an additional
    //            constrained problem built on top of A*x = b.
    //
    //   x      : original/primal degrees of freedom.
    //
    //   lambda : constraint multiplier degrees of freedom. These are auxiliary
    //            unknowns introduced to impose the constraint equations.
    //
    //   G      : multiplier-to-primal coupling block. It contributes lambda
    //            sensitivities/corrections to the original x-equations; i.e.
    //            rows associated with x and columns associated with lambda.
    //
    //   H      : primal-to-constraint coupling block. It contributes x
    //            sensitivities to the constraint equations; i.e. rows associated
    //            with constraints and columns associated with x.
    //
    //   M      : multiplier-to-constraint block. It couples lambda DOFs to the
    //            constraint equations.
    //
    //   bc     : right-hand-side/residual vector of the constraint equations.
    //
    // Eliminating lambda gives the Schur-condensed system:
    //
    //     (A - G*M^{-1}*H)*x = b - G*M^{-1}*bc
    //
    // Therefore, Schur condensation modifies the original A and b so that the
    // solved system includes the effect of the imposed constraints without
    // keeping lambda as an active unknown in the final linear solve.
    struct SchurData
    {
        using GlobalNodeId = std::uint64_t;
        static constexpr std::size_t NSq = N * N;
        using Block = std::array<DataType, NSq>;

        std::map<GlobalNodeId, std::map<int, Block>> G;
        std::vector<std::map<GlobalNodeId, Block>> H;
        Vector M;     // numC * NSq
        Vector bc;    // numC * N

        // Cached by condenseSchur(), consumed by calculateLambda().
        Vector Minv;
        Vector MinvBc;
        std::vector<std::map<GlobalNodeId, Block>> MinvH;
        std::vector<char> lambdaActive;

        // Populated by calculateLambda().
        Vector lambda; // numC * N

        // Block-level fragment writers
        Block& GBlock(GlobalNodeId row, int c) { return G[row][c]; }
        Block& HBlock(int c, GlobalNodeId col)
        {
            assert(c >= 0 && static_cast<std::size_t>(c) < H.size());
            return H[static_cast<std::size_t>(c)][col];
        }
        DataType* DBlock(int c)
        {
            assert(c >= 0 &&
                   static_cast<std::size_t>(c) * NSq < M.size());
            return &M[static_cast<std::size_t>(c) * NSq];
        }
        DataType* bcBlock(int c)
        {
            assert(c >= 0 &&
                   static_cast<std::size_t>(c) * N < bc.size());
            return &bc[static_cast<std::size_t>(c) * N];
        }
    };

    SchurData& getSchurData() { return schur_data_; }
    const SchurData& getSchurData() const { return schur_data_; }

    // Idempotent sizing helper. Call at the start of every assemble
    // pass before the per-fragment scatter runs.
    void resizeSchurCoefficients(std::size_t numC)
    {
        schur_data_.G.clear();
        schur_data_.H.assign(numC, {});
        schur_data_.M.assign(numC * SchurData::NSq, 0.0);
        schur_data_.bc.assign(numC * BLOCKSIZE, 0.0);
    }

    // Recovered seam value at cId, component ic. Returns 0 for c
    // slots that were not active during the most recent
    // condenseSchur() / calculateLambda() pass.
    DataType lambda(int cId, std::size_t ic) const
    {
        assert(cId >= 0);
        const std::size_t idx =
            static_cast<std::size_t>(cId) * BLOCKSIZE + ic;
        if (idx >= schur_data_.lambda.size())
            return DataType(0);
        return schur_data_.lambda[idx];
    }

    std::size_t lambdaSize() const
    {
        return schur_data_.lambda.size() / BLOCKSIZE;
    }

    // Cross-rank merges M, bc, H; pre-computes per-c Minv,
    // MinvBc, and per-(c, col) MinvH; then sums the per-(row, c)
    // Schur deltas into the matrix A and right-hand side b for every
    // locally-owned row that the assembly pass touched.
    template <typename GlobalToLocal>
    void condenseSchur(GlobalToLocal&& globalToLocal);

    // Recover the Lagrange-multiplier seam values from the solved
    // x vector.
    template <typename GlobalToLocal>
    void calculateLambda(GlobalToLocal&& globalToLocal);

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

    void serialize(const std::string basename = "coeffs") const
    {
        const int rank = this->commRank();
        const int size = this->commSize();
        std::ostringstream fname;
        fname << basename << '.' << rank << '.' << size;
        std::ofstream fout(fname.str(), std::ios::binary);

        // graph
        this->graph_->serialize(fout);

        // coefficients (always write 64-bit)
        uint64_t v64;
        double fp64;

        // header
        const char* p64 = reinterpret_cast<char*>(&v64);
        v64 = BLOCKSIZE;
        fout.write(p64, sizeof(uint64_t));
        v64 = this->values_.size();
        fout.write(p64, sizeof(uint64_t));
        v64 = this->b_.size();
        fout.write(p64, sizeof(uint64_t));

        // data
        p64 = reinterpret_cast<char*>(&fp64);
        for (const DataType v : this->values_)
        {
            fp64 = v;
            fout.write(p64, sizeof(double));
        }
        for (const DataType v : this->b_)
        {
            fp64 = v;
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

    // GGI constrained-mortar augmented-system storage + caches.
    // Sized lazily via resizeSchurCoefficients(numC); empty for
    // assembly passes that do not use CM.
    SchurData schur_data_;
};

} /* namespace linearSolver */

// schur-complement operations (wrapper bodies)

#include "matrix/CRSNodeGraph.h"
#include "matrix/operators/schur/schurCondense.h"
#include "matrix/operators/schur/schurLambdaRecovery.h"

namespace linearSolver
{

template <size_t N>
template <typename GlobalToLocal>
void coefficients<N>::condenseSchur(GlobalToLocal&& globalToLocal)
{
    schur::condense<N>(*static_cast<Matrix*>(this),
                       b_,
                       schur_data_,
                       this->getCommunicator(),
                       *this->getGraph(),
                       std::forward<GlobalToLocal>(globalToLocal));
}

template <size_t N>
template <typename GlobalToLocal>
void coefficients<N>::calculateLambda(GlobalToLocal&& globalToLocal)
{
    schur::calculateLambda<N>(x_,
                             schur_data_,
                             this->getCommunicator(),
                             std::forward<GlobalToLocal>(globalToLocal));
}

} /* namespace linearSolver */

#endif /* COEFFICIENTS_H_E818A4KY */
