// File       : schurLambdaRecovery.h
// Description: GGI constrained-mortar schur Lagrange-multiplier
//              recovery operator. After the linear solve has written
//              into x, computes lambda = M^{-1} (bc - H * x)
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef SCHURPSIRECOVERY_H_LWS3K2MX
#define SCHURPSIRECOVERY_H_LWS3K2MX

#include "blockMatrixOperators.h"
#include "matrix/CRSMatrix.h"
#include <cstddef>
#include <mpi.h>
#include <vector>

namespace linearSolver
{
namespace schur
{

// COLLECTIVE on `comm` — every rank must call.
template <size_t N, typename SchurData, typename GlobalToLocal>
void calculateLambda(const typename CRSMatrix<N>::Vector& x,
                    SchurData& sd,
                    MPI_Comm comm,
                    GlobalToLocal&& globalToLocal)
{
    using DataType = typename CRSMatrix<N>::DataType;
    using Index = typename CRSMatrix<N>::Index;
    constexpr int NB = static_cast<int>(N);

    const std::size_t numC = sd.M.size() / SchurData::NSq;
    if (numC == 0 || sd.MinvBc.empty())
    {
        return;
    }

    // Start from MinvBc (= Minv * bc).
    sd.lambda.assign(numC * N, DataType(0));
    for (std::size_t c = 0; c < numC; ++c)
    {
        if (!sd.lambdaActive[c])
        {
            continue;
        }
        for (std::size_t ic = 0; ic < N; ++ic)
        {
            sd.lambda[c * N + ic] = sd.MinvBc[c * N + ic];
        }
    }

    // Subtract sum_col MinvH[c, col] * x[col]. Each col is an
    // off-face node whose value lives in x at offset
    // local_id(col) * N + ic. The caller-provided globalToLocal
    // callable resolves the global → local index that matches the
    // matrix graph's indexing.
    for (std::size_t c = 0; c < numC; ++c)
    {
        if (!sd.lambdaActive[c])
        {
            continue;
        }
        for (const auto& colEntry : sd.MinvH[c])
        {
            const typename SchurData::GlobalNodeId colGid =
                colEntry.first;
            const auto& MinvHBlock = colEntry.second;

            const Index colLid = globalToLocal(colGid);
            if (colLid < 0)
            {
                continue;
            }

            DataType xvec[N];
            for (std::size_t jc = 0; jc < N; ++jc)
            {
                xvec[jc] = x[colLid * N + jc];
            }

            // lambda[c] -= MinvH[c, col] * xvec
            BlockMatrix::matrixVectorSub<NB>(MinvHBlock.data(),
                                             xvec,
                                             &sd.lambda[c * N]);
        }
    }

    // Cross-rank merge: each rank contributes its locally-visible
    // (MinvH · x) sum; MinvBc / MinvH are identical across ranks
    // (already merged in condense). Allreduce-SUM the partial
    // subtraction so every rank ends with the same lambda.
    int nproc = 1;
    MPI_Comm_size(comm, &nproc);
    if (nproc > 1)
    {
        std::vector<DataType> tmp(numC * N);
        for (std::size_t i = 0; i < numC * N; ++i)
        {
            tmp[i] = sd.lambda[i] - sd.MinvBc[i];
        }
        std::vector<DataType> reduced(numC * N, DataType(0));
        MPI_Allreduce(tmp.data(),
                      reduced.data(),
                      static_cast<int>(numC * N),
                      MPI_DOUBLE,
                      MPI_SUM,
                      comm);
        for (std::size_t i = 0; i < numC * N; ++i)
        {
            sd.lambda[i] = sd.MinvBc[i] + reduced[i];
        }
    }
}

} /* namespace Schur */
} /* namespace linearSolver */

#endif /* SCHURPSIRECOVERY_H_LWS3K2MX */
