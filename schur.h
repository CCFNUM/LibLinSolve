// File       : schur.h
// Created    : Tue May 19 2026
// Author     : Mhamad Mahdi Alloush
// Description: Schur-complement operations
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef SCHUR_H_LWS3K2MX
#define SCHUR_H_LWS3K2MX

#include "blockMatrixOperators.h"
#include "coefficients.h"
#include "CRSMatrix.h"
#include "CRSNodeGraph.h"
#include <cmath>
#include <cstdint>
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace linearSolver
{
namespace schur
{

// COLLECTIVE on `comm` — every rank must call.
template <size_t N, typename GlobalToLocalRow, typename GlobalToColumn>
void condense(coefficients<N>& coeffs,
              GlobalToLocalRow&& globalToLocalRow,
              GlobalToColumn&& globalToColumn)
{
    using SchurData = typename coefficients<N>::SchurData;
    using DataType = typename CRSMatrix<N>::DataType;
    using Index = typename CRSMatrix<N>::Index;
    constexpr int NB = static_cast<int>(N);

    CRSMatrix<N>& A = coeffs;
    auto& b = coeffs.getBVector();
    SchurData& sd = coeffs.getSchurData();
    const MPI_Comm comm = coeffs.getCommunicator();
    const CRSNodeGraph& graph = *coeffs.getGraph();

    const Index numC = sd.M.size() / SchurData::NSq;
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
            std::vector<DataType> Mlocal = sd.M;
            MPI_Allreduce(Mlocal.data(),
                          sd.M.data(),
                          static_cast<int>(sd.M.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          comm);
            std::vector<DataType> bcLocal = sd.bc;
            MPI_Allreduce(bcLocal.data(),
                          sd.bc.data(),
                          static_cast<int>(sd.bc.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          comm);

            // Pack local H as (c, col, NSq doubles) triples.
            std::vector<Index> triCs;
            std::vector<Index> triCol;
            std::vector<DataType> triVal;
            triCs.reserve(numC);
            triCol.reserve(numC);
            for (Index c = 0; c < numC; ++c)
            {
                for (const auto& [col, blk] : sd.H[c])
                {
                    triCs.push_back(c);
                    triCol.push_back(col);
                    for (Index k = 0; k < SchurData::NSq; ++k)
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
            std::vector<Index> allCs(total);
            std::vector<Index> allCol(total);
            std::vector<DataType> allVal(total * SchurData::NSq);
            MPI_Allgatherv(triCs.data(),
                           mineCount,
                           MPIDataType<Index>::type(),
                           allCs.data(),
                           counts.data(),
                           displs.data(),
                           MPIDataType<Index>::type(),
                           comm);
            MPI_Allgatherv(triCol.data(),
                           mineCount,
                           MPIDataType<Index>::type(),
                           allCol.data(),
                           counts.data(),
                           displs.data(),
                           MPIDataType<Index>::type(),
                           comm);
            MPI_Allgatherv(triVal.data(),
                           mineCount * static_cast<int>(SchurData::NSq),
                           MPIDataType<DataType>::type(),
                           allVal.data(),
                           countsVals.data(),
                           displsVals.data(),
                           MPIDataType<DataType>::type(),
                           comm);
            for (auto& s : sd.H)
            {
                s.clear();
            }
            for (int i = 0; i < total; ++i)
            {
                auto& dst = sd.H[allCs[i]][allCol[i]];
                for (Index k = 0; k < SchurData::NSq; ++k)
                {
                    dst[k] += allVal[i * SchurData::NSq + k];
                }
            }

            // Cross-rank merge of G (A_cv)
            std::vector<Index> gRow;
            std::vector<Index> gCs;
            std::vector<DataType> gVal;
            for (const auto& [rowGid, csMap] : sd.G)
            {
                for (const auto& [cs, blk] : csMap)
                {
                    gRow.push_back(rowGid);
                    gCs.push_back(cs);
                    for (Index k = 0; k < SchurData::NSq; ++k)
                    {
                        gVal.push_back(blk[k]);
                    }
                }
            }
            const int gMine = static_cast<int>(gRow.size());
            std::vector<int> gCounts(nproc, 0);
            MPI_Allgather(
                &gMine, 1, MPI_INT, gCounts.data(), 1, MPI_INT, comm);
            std::vector<int> gDispls(nproc, 0);
            int gTotal = 0;
            for (int r = 0; r < nproc; ++r)
            {
                gDispls[r] = gTotal;
                gTotal += gCounts[r];
            }
            std::vector<int> gCountsVals(nproc, 0);
            std::vector<int> gDisplsVals(nproc, 0);
            for (int r = 0; r < nproc; ++r)
            {
                gCountsVals[r] =
                    gCounts[r] * static_cast<int>(SchurData::NSq);
                gDisplsVals[r] =
                    gDispls[r] * static_cast<int>(SchurData::NSq);
            }
            std::vector<Index> gAllRow(gTotal);
            std::vector<Index> gAllCs(gTotal);
            std::vector<DataType> gAllVal(gTotal * SchurData::NSq);
            MPI_Allgatherv(gRow.data(),
                           gMine,
                           MPIDataType<Index>::type(),
                           gAllRow.data(),
                           gCounts.data(),
                           gDispls.data(),
                           MPIDataType<Index>::type(),
                           comm);
            MPI_Allgatherv(gCs.data(),
                           gMine,
                           MPIDataType<Index>::type(),
                           gAllCs.data(),
                           gCounts.data(),
                           gDispls.data(),
                           MPIDataType<Index>::type(),
                           comm);
            MPI_Allgatherv(gVal.data(),
                           gMine * static_cast<int>(SchurData::NSq),
                           MPIDataType<DataType>::type(),
                           gAllVal.data(),
                           gCountsVals.data(),
                           gDisplsVals.data(),
                           MPIDataType<DataType>::type(),
                           comm);
            sd.G.clear();
            for (Index i = 0; i < gTotal; ++i)
            {
                auto& dst = sd.G[gAllRow[i]][gAllCs[i]];
                for (Index k = 0; k < SchurData::NSq; ++k)
                {
                    dst[k] += gAllVal[i * SchurData::NSq + k];
                }
            }
        }
    }

    // Pre-compute per-c Minv and MinvBc (persistent)
    sd.Minv.assign(numC * SchurData::NSq, DataType(0));
    sd.MinvBc.assign(numC * N, DataType(0));
    sd.cActive.assign(numC, 0);
    for (Index c = 0; c < numC; ++c)
    {
        const DataType* Mblock = &sd.M[c * SchurData::NSq];
        if (BlockMatrix::isZero<NB>(Mblock))
        {
            continue;
        }
        sd.cActive[c] = 1;
        const DataType det = BlockMatrix::determinant<NB>(Mblock);
        if (std::abs(det) == DataType(0))
        {
            throw std::runtime_error(
                "linearSolver::schur::condense: singular M block");
        }
        BlockMatrix::invert<NB>(Mblock, &sd.Minv[c * SchurData::NSq]);
        BlockMatrix::matrixVector<NB>(&sd.Minv[c * SchurData::NSq],
                                      &sd.bc[c * N],
                                      &sd.MinvBc[c * N]);
    }

    // MinvH[c][col] = Minv[c] * H[c, col]
    sd.MinvH.assign(numC, {});
    for (Index c = 0; c < numC; ++c)
    {
        if (!sd.cActive[c])
        {
            continue;
        }
        const DataType* MinvBlock = &sd.Minv[c * SchurData::NSq];
        for (const auto& [col, Hblock] : sd.H[c])
        {
            auto& dst = sd.MinvH[c][col];
            BlockMatrix::matrixMatrix<NB>(MinvBlock,
                                          Hblock.data(),
                                          dst.data());
        }
    }

    // Pre-allocate the recovery buffer (filled by calculateLambda).
    sd.lambda.assign(numC * N, DataType(0));

    // Apply Schur correction per (mesh row i, c)
    const auto nOwnedRows = graph.nOwnedNodes();

    for (const auto& rowEntry : sd.G)
    {
        const Index rowGid = rowEntry.first;

        const Index rowID = globalToLocalRow(rowGid);
        if (rowID < 0 || rowID >= nOwnedRows)
        {
            continue;
        }

        auto rowVals = A.rowVals(rowID);
        const auto rowCols = A.rowCols(rowID);
        const Index length = static_cast<Index>(rowCols.size());

        for (const auto& csEntry : rowEntry.second)
        {
            const Index c = csEntry.first;
            if (c < 0 || c >= numC)
            {
                continue;
            }
            if (!sd.cActive[c])
            {
                continue;
            }
            const auto& Gblock = csEntry.second;
            if (BlockMatrix::isZero<NB>(Gblock.data()))
            {
                continue;
            }

            // RHS contribution: b -= G * MinvBc
            DataType rhsDelta[N];
            BlockMatrix::matrixVector<NB>(Gblock.data(),
                                          &sd.MinvBc[c * N],
                                          rhsDelta);
            for (size_t ic = 0; ic < N; ++ic)
            {
                b[rowID * N + ic] -= rhsDelta[ic];
            }

            // Matrix contributions per col: A[row, col] -= G * MinvH[col]
            for (const auto& colEntry : sd.MinvH[c])
            {
                const Index colGid = colEntry.first;
                const auto& MinvHBlock = colEntry.second;
                DataType Adelta[SchurData::NSq];
                BlockMatrix::matrixMatrix<NB>(Gblock.data(),
                                              MinvHBlock.data(),
                                              Adelta);

                // Resolve the column key in the matrix's storage order.
                // The caller maps to local id (local-column-order graphs)
                // or renumbered global id (global-column-order graphs).
                const Index colKey = globalToColumn(colGid);
                if (colKey < 0)
                {
                    continue;
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

                // Subtract Adelta (= G * MinvH) from the (offset)-th
                // block of rowVals. No existing CRSMatrix-row-block-
                // scatter helper, so the index arithmetic stays inline.
                DataType* const blockDst =
                    &rowVals[SchurData::NSq * offset];
                BlockMatrix::matrixSubInplace<NB>(Adelta, blockDst);
            }
        }
    }

    // Zero the per-pass scatter buffers (G/H/M/bc) now that they
    // have been consumed.
    sd.G.clear();
    for (auto& hSlot : sd.H)
    {
        hSlot.clear();
    }
    std::fill(sd.M.begin(), sd.M.end(), DataType(0));
    std::fill(sd.bc.begin(), sd.bc.end(), DataType(0));
}

// COLLECTIVE on `comm` — every rank must call.
template <size_t N, typename GlobalToLocal>
void calculateLambda(coefficients<N>& coeffs, GlobalToLocal&& globalToLocal)
{
    using SchurData = typename coefficients<N>::SchurData;
    using DataType = typename CRSMatrix<N>::DataType;
    using Index = typename CRSMatrix<N>::Index;
    constexpr int NB = static_cast<int>(N);

    const auto& x = coeffs.getXVector();
    SchurData& sd = coeffs.getSchurData();
    const MPI_Comm comm = coeffs.getCommunicator();

    const Index numC = sd.M.size() / SchurData::NSq;
    if (numC == 0 || sd.MinvBc.empty())
    {
        return;
    }

    // Start from MinvBc (= Minv * bc).
    sd.lambda.assign(numC * N, DataType(0));
    for (Index c = 0; c < numC; ++c)
    {
        if (!sd.cActive[c])
        {
            continue;
        }
        for (size_t ic = 0; ic < N; ++ic)
        {
            sd.lambda[c * N + ic] = sd.MinvBc[c * N + ic];
        }
    }

    // Subtract sum_col MinvH[c, col] * x[col]. Each col is an
    // off-face node whose value lives in x at offset
    // local_id(col) * N + ic. The caller-provided globalToLocal
    // callable resolves the global → local index that matches the
    // matrix graph's indexing.
    for (Index c = 0; c < numC; ++c)
    {
        if (!sd.cActive[c])
        {
            continue;
        }
        for (const auto& colEntry : sd.MinvH[c])
        {
            const Index colGid = colEntry.first;
            const auto& MinvHBlock = colEntry.second;

            const Index colLid = globalToLocal(colGid);
            if (colLid < 0)
            {
                continue;
            }

            DataType xvec[N];
            for (size_t jc = 0; jc < N; ++jc)
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
        for (Index i = 0; i < static_cast<Index>(numC * N); ++i)
        {
            tmp[i] = sd.lambda[i] - sd.MinvBc[i];
        }
        std::vector<DataType> reduced(numC * N, DataType(0));
        MPI_Allreduce(tmp.data(),
                      reduced.data(),
                      static_cast<int>(numC * N),
                      MPIDataType<DataType>::type(),
                      MPI_SUM,
                      comm);
        for (Index i = 0; i < static_cast<Index>(numC * N); ++i)
        {
            sd.lambda[i] = sd.MinvBc[i] + reduced[i];
        }
    }
}

} /* namespace schur */
} /* namespace linearSolver */

#endif /* SCHUR_H_LWS3K2MX */
