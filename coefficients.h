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
#include <memory>
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
    using VectorHost = typename Matrix::VectorHost;
    using Index = typename Matrix::Index;
    using IndexVector = typename Matrix::IndexVector;
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
        resizeDataStructures();
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
        resizeDataStructures();
    }

    void resizeDataStructures()
    {
        const Index n_local_coeff = graph_.nOwnedNodes();
        const Index n_local_ghosts = graph_.nGhostNodes();
        assert(graph_.isBuilt()); // MPI data is correct if graph is
                                  // built

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
        static constexpr Index NSq = N * N;
        using Block = std::array<DataType, NSq>;

        std::map<Index, std::map<Index, Block>> G;
        std::vector<std::map<Index, Block>> H;
        std::vector<DataType> M;  // numC * NSq
        std::vector<DataType> bc; // numC * N

        std::vector<DataType> Minv;
        std::vector<DataType> MinvBc;
        std::vector<std::map<Index, Block>> MinvH;
        std::vector<char> cActive;
        std::vector<DataType> lambda; // numC * N

        // Block-level writers
        Block& GBlock(const Index row, const Index c)
        {
            return G[row][c];
        }

        Block& HBlock(const Index c, const Index col)
        {
            assert(c >= 0 && static_cast<size_t>(c) < H.size());
            return H[c][col];
        }

        DataType* MBlock(const Index c)
        {
            assert(c >= 0 && static_cast<size_t>(c) * NSq < M.size());
            return &M[c * NSq];
        }

        DataType* bcBlock(const Index c)
        {
            assert(c >= 0 && static_cast<size_t>(c) * N < bc.size());
            return &bc[c * N];
        }
    };

    SchurData& getSchurData() { return schur_data_; }
    const SchurData& getSchurData() const { return schur_data_; }

    // Idempotent sizing helper
    void resizeSchurCoefficients(const size_t numC)
    {
        if (numConstraints() == numC)
            return;
        schur_data_.G.clear();
        schur_data_.H.assign(numC, {});
        schur_data_.M.assign(numC * SchurData::NSq, 0.0);
        schur_data_.bc.assign(numC * BLOCKSIZE, 0.0);
    }

    // Number of constraint equations carried by the augmented system —
    // equivalently, the number of multiplier (lambda) slots.
    size_t numConstraints() const
    {
        return schur_data_.H.size();
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

private:
    const Index id_; // coefficient container ID

    Vector x_; // unknowns
    Vector b_; // right-hand-side
    Vector r_; // residual vector r = b - Ax

    // auxiliary data (not initialized nor resized during coefficient
    // construction since only required for specific solvers)
    LUData LU_data_;

    // augmented-system storage + caches.
    // Sized lazily via resizeSchurCoefficients(numC)
    SchurData schur_data_;
};

} /* namespace linearSolver */

#endif /* COEFFICIENTS_H_E818A4KY */
