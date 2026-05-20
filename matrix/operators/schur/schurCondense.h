// File       : schurCondense.h
// Created    : Tue May 19 2026
// Author     : Mhamad Mahdi Alloush
// Description: GGI constrained-mortar schur-complement condensation
//              operator.
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef SCHURCONDENSE_H_LWS3K2MX
#define SCHURCONDENSE_H_LWS3K2MX

#include "blockMatrixOperators.h"
#include "matrix/CRSMatrix.h"
#include "matrix/CRSNodeGraph.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace linearSolver
{
namespace schur
{

// COLLECTIVE on `comm` — every rank must call.
template <size_t N, typename SchurData, typename GlobalToLocal>
void condense(CRSMatrix<N>& A,
              typename CRSMatrix<N>::Vector& b,
              SchurData& sd,
              MPI_Comm comm,
              const CRSNodeGraph& graph,
              GlobalToLocal&& globalToLocal)
{
    using DataType = typename CRSMatrix<N>::DataType;
    using Index = typename CRSMatrix<N>::Index;
    constexpr int NB = static_cast<int>(N);

    const std::size_t numC = sd.M.size() / SchurData::NSq;
    if (numC == 0)
    {
        return;
    }

    // Cross-rank merge
    {
        int nproc = 1;
        MPI_Comm_size(comm, &nproc);

        if (nproc > 1)
        {
            std::vector<DataType> Dlocal = sd.M;
            MPI_Allreduce(Dlocal.data(),
                          sd.M.data(),
                          static_cast<int>(sd.M.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          comm);
            std::vector<DataType> rcsLocal = sd.bc;
            MPI_Allreduce(rcsLocal.data(),
                          sd.bc.data(),
                          static_cast<int>(sd.bc.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          comm);

            // Pack local H as (c, col, NSq doubles) triples.
            std::vector<int> triCs;
            std::vector<std::uint64_t> triCol;
            std::vector<DataType> triVal;
            triCs.reserve(numC);
            triCol.reserve(numC);
            for (std::size_t c = 0; c < numC; ++c)
            {
                for (const auto& [col, blk] : sd.H[c])
                {
                    triCs.push_back(static_cast<int>(c));
                    triCol.push_back(col);
                    for (std::size_t k = 0; k < SchurData::NSq; ++k)
                    {
                        triVal.push_back(blk[k]);
                    }
                }
            }
            const int mineCount = static_cast<int>(triCs.size());
            std::vector<int> counts(nproc, 0);
            MPI_Allgather(
                &mineCount, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
            std::vector<int> displs(nproc, 0);
            int total = 0;
            for (int r = 0; r < nproc; ++r)
            {
                displs[r] = total;
                total += counts[r];
            }
            std::vector<int> countsVals(nproc, 0);
            std::vector<int> displsVals(nproc, 0);
            for (int r = 0; r < nproc; ++r)
            {
                countsVals[r] =
                    counts[r] * static_cast<int>(SchurData::NSq);
                displsVals[r] =
                    displs[r] * static_cast<int>(SchurData::NSq);
            }
            std::vector<int> allCs(total);
            std::vector<std::uint64_t> allCol(total);
            std::vector<DataType> allVal(
                static_cast<std::size_t>(total) * SchurData::NSq);
            MPI_Allgatherv(triCs.data(),
                           mineCount,
                           MPI_INT,
                           allCs.data(),
                           counts.data(),
                           displs.data(),
                           MPI_INT,
                           comm);
            MPI_Allgatherv(triCol.data(),
                           mineCount,
                           MPI_UINT64_T,
                           allCol.data(),
                           counts.data(),
                           displs.data(),
                           MPI_UINT64_T,
                           comm);
            MPI_Allgatherv(triVal.data(),
                           mineCount * static_cast<int>(SchurData::NSq),
                           MPI_DOUBLE,
                           allVal.data(),
                           countsVals.data(),
                           displsVals.data(),
                           MPI_DOUBLE,
                           comm);
            for (auto& s : sd.H)
            {
                s.clear();
            }
            for (int i = 0; i < total; ++i)
            {
                auto& dst =
                    sd.H[static_cast<std::size_t>(allCs[i])][allCol[i]];
                for (std::size_t k = 0; k < SchurData::NSq; ++k)
                {
                    dst[k] +=
                        allVal[static_cast<std::size_t>(i) * SchurData::NSq +
                               k];
                }
            }
        }
    }

    // Pre-compute per-c Minv and MinvBc (persistent)
    sd.Minv.assign(numC * SchurData::NSq, DataType(0));
    sd.MinvBc.assign(numC * N, DataType(0));
    sd.lambdaActive.assign(numC, 0);
    for (std::size_t c = 0; c < numC; ++c)
    {
        const DataType* Dblock = &sd.M[c * SchurData::NSq];
        if (BlockMatrix::isZero<NB>(Dblock))
        {
            continue;
        }
        sd.lambdaActive[c] = 1;
        const DataType det = BlockMatrix::determinant<NB>(Dblock);
        if (std::abs(det) == DataType(0))
        {
            throw std::runtime_error(
                "linearSolver::Schur::condense: singular M block");
        }
        BlockMatrix::invert<NB>(Dblock, &sd.Minv[c * SchurData::NSq]);
        BlockMatrix::matrixVector<NB>(&sd.Minv[c * SchurData::NSq],
                                      &sd.bc[c * N],
                                      &sd.MinvBc[c * N]);
    }

    // MinvH[c][col] = Minv[c] * H[c, col]
    sd.MinvH.assign(numC, {});
    for (std::size_t c = 0; c < numC; ++c)
    {
        if (!sd.lambdaActive[c])
        {
            continue;
        }
        const DataType* DinvBlock = &sd.Minv[c * SchurData::NSq];
        for (const auto& [col, AscBlock] : sd.H[c])
        {
            auto& dst = sd.MinvH[c][col];
            BlockMatrix::matrixMatrix<NB>(DinvBlock,
                                          AscBlock.data(),
                                          dst.data());
        }
    }

    // Pre-allocate the recovery buffer (filled by calculateLambda).
    sd.lambda.assign(numC * N, DataType(0));

    // Apply Schur correction per (mesh row i, c)
    const bool localColumnOrder = graph.isLocalColumnOrder();
    const auto nOwnedRows = graph.nOwnedNodes();

    for (const auto& rowEntry : sd.G)
    {
        const typename SchurData::GlobalNodeId rowGid = rowEntry.first;

        const Index rowID = globalToLocal(rowGid);
        if (rowID < 0 || rowID >= nOwnedRows)
        {
            continue;
        }

        auto rowVals = A.rowVals(rowID);
        const auto rowCols = A.rowCols(rowID);
        const Index length = static_cast<Index>(rowCols.size());

        for (const auto& csEntry : rowEntry.second)
        {
            const int c = csEntry.first;
            if (c < 0 || static_cast<std::size_t>(c) >= numC)
            {
                continue;
            }
            if (!sd.lambdaActive[c])
            {
                continue;
            }
            const auto& Acv = csEntry.second;
            if (BlockMatrix::isZero<NB>(Acv.data()))
            {
                continue;
            }

            // RHS contribution: b += -G * MinvBc
            DataType rhsDelta[N];
            BlockMatrix::matrixVector<NB>(Acv.data(),
                                          &sd.MinvBc[c * N],
                                          rhsDelta);
            for (std::size_t ic = 0; ic < N; ++ic)
            {
                b[rowID * N + ic] -= rhsDelta[ic];
            }

            // Matrix contributions per col: A[row, col] += -G * MinvH[col]
            for (const auto& colEntry : sd.MinvH[c])
            {
                const typename SchurData::GlobalNodeId colGid =
                    colEntry.first;
                const auto& DinvHBlock = colEntry.second;
                DataType M[SchurData::NSq];
                BlockMatrix::matrixMatrix<NB>(Acv.data(),
                                              DinvHBlock.data(),
                                              M);

                Index colKey;
                if (localColumnOrder)
                {
                    const Index colLid = globalToLocal(colGid);
                    if (colLid < 0)
                    {
                        continue;
                    }
                    colKey = colLid;
                }
                else
                {
                    colKey = static_cast<Index>(colGid);
                }

                // Walk the row to find the column offset.
                Index offset = 0;
                while (offset < length && rowCols[offset] != colKey)
                {
                    ++offset;
                }
                if (offset >= length)
                {
                    // Should not happen now that nodeGraph::buildGraph_
                    // pre-allocates every current-element x opposing-
                    // element pair AND every intra-element same-side
                    // pair for non-conformal interface elements.
                    continue;
                }

                // Subtract M from the (offset)-th block of rowVals.
                // No existing CRSMatrix-row-block-scatter helper, so
                // the index arithmetic stays inline.
                DataType* const blockDst =
                    &rowVals[SchurData::NSq * offset];
                BlockMatrix::matrixSubInplace<NB>(M, blockDst);
            }
        }
    }
}

} /* namespace Schur */
} /* namespace linearSolver */

#endif /* SCHURCONDENSE_H_LWS3K2MX */
