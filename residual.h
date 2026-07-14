// File       : residual.h
// Created    : Sun Oct 13 2024 18:48:31 (+0200)
// Author     : Fabian Wermelinger
// Description: Linear solver residual functions
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef RESIDUAL_H_SIKC1UDI
#define RESIDUAL_H_SIKC1UDI

#include "CRSResidual.h"
#include "coefficients.h"
#include "controlData.h"
#include "kokkosReductionUtils.hpp"
#include <cassert>
#include <cmath>
#include <mpi.h>

namespace linearSolver
{
namespace residual
{

template <size_t N>
using Array = typename ControlData<N>::Array;

template <size_t N>
constexpr void compute(coefficients<N>& coeff)
{
    using Matrix = typename coefficients<N>::Matrix;
    using Vector = typename coefficients<N>::Vector;

    const Matrix& A = coeff.getAMatrix();
    const Vector& b = coeff.getBVector();
    Vector& x = coeff.getXVector();
    Vector& r = coeff.getRVector();
    CRSResidual(A, b, x, r);
}

template <size_t N>
constexpr Array<N>
RMSReduceAndScale(const MPI_Comm comm, const Array<N>& rms, const double scale)
{
    Array<N> global_rms = {0};
    MPI_Allreduce(
        rms.data(), global_rms.data(), rms.size(), MPI_DOUBLE, MPI_SUM, comm);
    for (size_t k = 0; k < N; k++)
    {
        global_rms[k] = std::sqrt(global_rms[k] * scale);
    }

    return global_rms;
}

template <size_t N, typename VectorType, typename Index>
constexpr Array<N> RMSError(const MPI_Comm comm,
                            const Index n_local,
                            const Index n_global,
                            const VectorType& R)
{
    constexpr Index BLOCKSIZE = N;

    using ScalarType = VectorType::value_type;
    using ReduceValueType =
        NgpUtils::Reduction::FixedArray<ScalarType, BLOCKSIZE>;
    using ReducerType =
        NgpUtils::Reduction::FixedArraySum<ScalarType, BLOCKSIZE>;

    ReduceValueType rmsRed;
    ReducerType reducer(rmsRed);

    Kokkos::parallel_reduce(
        "RMSError",
        Kokkos::RangePolicy<AccelExecSpace>(0, n_local),
        KOKKOS_LAMBDA(const Index i, ReduceValueType& local_rms) {
            ReduceValueType resSq;
            for (Index k = 0; k < BLOCKSIZE; ++k)
            {
                resSq[k] = R[BLOCKSIZE * i + k] * R[BLOCKSIZE * i + k];
            }
            reducer.join(local_rms, resSq);
        },
        reducer);

    Kokkos::fence();

    Array<N> rms = {0}; // root-mean-square of residual vector

    for (Index k = 0; k < BLOCKSIZE; ++k)
    {
        rms[k] = rmsRed[k];
    }

    return RMSReduceAndScale<BLOCKSIZE>(comm, rms, 1.0 / n_global);
}

template <size_t N>
constexpr Array<N> RMSError(const coefficients<N>& coeff)
{
    const auto& R = coeff.getRVector();
    assert(coeff.getAMatrix().nRows() == coeff.nCoefficients());
    assert(R.size() <= N * coeff.nCoefficients());
    return RMSError<N>(coeff.getCommunicator(),
                       coeff.nCoefficients(),
                       coeff.nGlobalCoefficients(),
                       R);
}

template <size_t N>
constexpr Array<N> RMSErrorDiagNormalized(const coefficients<N>& coeff)
{
    using Matrix = typename coefficients<N>::Matrix;
    using Vector = typename coefficients<N>::Vector;
    using Index = typename coefficients<N>::Index;
    using DataType = typename Matrix::DataType;
    constexpr Index BLOCKSIZE = N;

    const Matrix& A = coeff.getAMatrix();
    const Vector& R = coeff.getRVector();

    assert(A.nRows() == coeff.nCoefficients());
    assert(static_cast<Index>(R.size()) <= BLOCKSIZE * coeff.nCoefficients());
    const Index nrows = coeff.nCoefficients(); // rank local rows

    using ReduceValueType =
        NgpUtils::Reduction::FixedArray<DataType, BLOCKSIZE>;
    using ReducerType = NgpUtils::Reduction::FixedArraySum<DataType, BLOCKSIZE>;

    ReduceValueType rmsRed;
    ReducerType reducer(rmsRed);

    Kokkos::parallel_reduce(
        "RMSErrorDiagNormalized",
        Kokkos::RangePolicy<AccelExecSpace>(0, nrows),
        KOKKOS_LAMBDA(const Index row, ReduceValueType& local_rms) {
            auto rowVals = A.blockRowVals(row);
            const typename Matrix::Index diagOffset = A.diagOffset(row);
            auto blockDiag = rowVals.block(diagOffset);

            ReduceValueType resSq;
            for (Index k = 0; k < BLOCKSIZE; ++k)
            {
                const DataType Ap = blockDiag(k, k);
                assert(Ap > static_cast<DataType>(0.0));
                const DataType r = R[BLOCKSIZE * row + k] / Ap;
                resSq[k] = r * r;
            }
            reducer.join(local_rms, resSq);
        },
        reducer);

    Kokkos::fence();

    Array<N> rms = {0}; // root-mean-square of residual vector

    for (Index k = 0; k < BLOCKSIZE; ++k)
    {
        rms[k] = rmsRed[k];
    }

    return RMSReduceAndScale<BLOCKSIZE>(
        coeff.getCommunicator(), rms, 1.0 / coeff.nGlobalCoefficients());
}

} /* namespace residual */
} /* namespace linearSolver */

#endif /* RESIDUAL_H_SIKC1UDI */
