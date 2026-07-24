// File       : CRSConnectivity.h
// Created    : Wed Oct 02 2024 11:37:50 (+0200)
// Author     : Fabian Wermelinger
// Description: Graph connectivity interface for matrices
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef CRSCONNECTIVITY_H_3ADJSEDT
#define CRSCONNECTIVITY_H_3ADJSEDT

#include "CRSNodeGraph.h"

#include <cassert>
#include <mpi.h>
#include <span>
#include <vector>

namespace linearSolver
{

class CRSConnectivity
{
public:
    using Index = CRSNodeGraph::Index;
    using IndexView = CRSNodeGraph::IndexView;
    using IndexSubview = CRSNodeGraph::IndexSubview;

    CRSConnectivity() = delete;

    CRSConnectivity(const CRSNodeGraph* graph) : graph_(*graph)
    {
    }

    virtual ~CRSConnectivity()
    {
    }

    KOKKOS_INLINE_FUNCTION
    const CRSNodeGraph& getGraph() const
    {
        return graph_;
    }

    MPI_Comm getCommunicator() const
    {
        return graph_.getCommunicator();
    }

    int commRank() const
    {
        return graph_.commRank();
    }

    int commSize() const
    {
        return graph_.commSize();
    }

    KOKKOS_INLINE_FUNCTION Index globalRowOffset() const
    {
        return graph_.globalRowOffset();
    };

    KOKKOS_INLINE_FUNCTION Index nRows() const
    {
        return graph_.nRows();
    };

    KOKKOS_INLINE_FUNCTION Index nGlobalRows() const
    {
        return graph_.nGlobalRows();
    };

    KOKKOS_INLINE_FUNCTION Index nnzBlocks() const
    {
        return graph_.nIndices();
    }

    KOKKOS_INLINE_FUNCTION unsigned long long nnzGlobalBlocks() const
    {
        return graph_.nGlobalIndices();
    }

    KOKKOS_INLINE_FUNCTION const RowIndex* offsetsPtr() const
    {
        return graph_.offsets().data();
    }

    KOKKOS_INLINE_FUNCTION const RowPtrView& offsetsRef() const
    {
        return graph_.offsets();
    }

    KOKKOS_INLINE_FUNCTION const Index* indicesPtr() const
    {
        return graph_.indices().data();
    }

    KOKKOS_INLINE_FUNCTION const IndexView& indicesRef() const
    {
        return graph_.indices();
    }

    KOKKOS_INLINE_FUNCTION const Index* diagOffsetPtr() const
    {
        return graph_.diagonalIndicesOffset().data();
    }

    KOKKOS_INLINE_FUNCTION const IndexView& diagOffsetRef() const
    {
        return graph_.diagonalIndicesOffset();
    }

    KOKKOS_INLINE_FUNCTION Index diagOffset(Index iRow) const
    {
        return graph_.diagonalIndicesOffset()[iRow];
    }

    KOKKOS_INLINE_FUNCTION IndexSubview rowCols(Index iRow) const
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        const auto& row_ptr = this->offsetsRef();
        return Kokkos::subview(
            this->indicesRef(),
            Kokkos::make_pair(row_ptr[iRow], row_ptr[iRow + 1]));
    }

    KOKKOS_INLINE_FUNCTION Index localToGlobal(Index localID) const
    {
        return graph_.localToGlobalIndex(localID);
    };

    KOKKOS_INLINE_FUNCTION Index globalToLocal(Index globalID) const
    {
        return graph_.globalToLocalIndex(globalID);
    };

protected:
    const CRSNodeGraph graph_;
};

} // namespace linearSolver

#endif /* CRSCONNECTIVITY_H_3ADJSEDT */
