// File       : CRSNodeGraph.h
// Created    : Sat Apr 13 2024 23:13:14 (+0200)
// Author     : Mhamad Mahdi Alloush, Fabian Wermelinger
// Description: Node graph generator base class
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef CRSNODEGRAPH_H_UI1807EK
#define CRSNODEGRAPH_H_UI1807EK

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mpi.h>
#include <span>
#include <vector>

#include "linearSolverTypes.h"

namespace linearSolver
{

// MPI type handles
template <typename T>
struct MPIDataType
{
    static inline MPI_Datatype type()
    {
        return MPI_DATATYPE_NULL;
    }
};

template <>
struct MPIDataType<double>
{
    static inline MPI_Datatype type()
    {
        return MPI_DOUBLE;
    }
};

template <>
struct MPIDataType<float>
{
    static inline MPI_Datatype type()
    {
        return MPI_FLOAT;
    }
};

template <>
struct MPIDataType<unsigned int>
{
    static inline MPI_Datatype type()
    {
        return MPI_UNSIGNED;
    }
};

template <>
struct MPIDataType<int32_t>
{
    static inline MPI_Datatype type()
    {
        return MPI_INT32_T;
    }
};

template <>
struct MPIDataType<int64_t>
{
    static inline MPI_Datatype type()
    {
        return MPI_INT64_T;
    }
};

struct MemoryFootprint
{
    static constexpr char UNIT[] = "MB";
    static constexpr unsigned int BYTE_DIVIDE = (1 << 20); // MB
    unsigned long long int sum_byte;
};

class GraphLayout
{
public:
    enum : int
    {
        // identifiers must be powers of two
        //
        // graph layout
        ColumnIndexOrder__Local = 0x001,
        ColumnIndexOrder__Global = 0x002,
        Stencil__Reduced = 0x004,
        // memory layout
        MemoryLayout__ConvertInplace = 0x008,
        MemoryLayout__BlockRowMajor = 0x010,
        MemoryLayout__RowMajor = 0x020,
        // convenience
        MemoryLayout__PETScAIJ = MemoryLayout__RowMajor,
        MemoryLayout__HYPRE_IJ = MemoryLayout__RowMajor,
        MemoryLayout__AllBits =
            MemoryLayout__BlockRowMajor | MemoryLayout__RowMajor,
        ColumnIndexOrder__AllBits =
            ColumnIndexOrder__Local | ColumnIndexOrder__Global,
        UNDEFINED = 0x000,
        Stencil__Full = UNDEFINED
    };

    KOKKOS_INLINE_FUNCTION
    GraphLayout() : value_(UNDEFINED)
    {
    }

    KOKKOS_INLINE_FUNCTION
    GraphLayout(const int c) : value_(c)
    {
    }

    KOKKOS_INLINE_FUNCTION
    operator int() const
    {
        return value_;
    }

private:
    int value_;
};

class CRSNodeGraph
{
public:
    // Main graph index type.  This type is used to compute differences and
    // should be a signed integral.
    using Index = TGraphIndex; // == ColIndex == staticcrsgraph_type::data_type
    using IndexView = ::IndexView;
    using IndexViewHost = ::IndexViewHost;
    using IndexSubview = ::IndexSubview;
    using IndexSubviewHost = ::IndexSubviewHost;

    // DAVEKOKKOS: move accelexecspace to global
    struct PackInfo
    {
        int remote_rank;
        std::vector<Index> send_idx;
        std::vector<Index> recv_idx;
    };

    CRSNodeGraph() = delete;

    explicit CRSNodeGraph(const MPI_Comm comm, const GraphLayout layout);

    virtual ~CRSNodeGraph() = default;

    void buildGraph();

    KOKKOS_INLINE_FUNCTION MPI_Comm getCommunicator() const
    {
        return comm_;
    }

    KOKKOS_INLINE_FUNCTION int commRank() const
    {
        return rank_;
    }

    KOKKOS_INLINE_FUNCTION int commSize() const
    {
        return size_;
    }

    KOKKOS_INLINE_FUNCTION Index nOwnedNodes() const
    {
        assert(is_built_);
        assert(n_owned_nodes_ > 0);
        return n_owned_nodes_;
    }

    KOKKOS_INLINE_FUNCTION Index nGhostNodes() const
    {
        assert(is_built_);
        return n_ghost_nodes_;
    }

    KOKKOS_INLINE_FUNCTION Index nAllNodes() const
    {
        assert(is_built_);
        return nOwnedNodes() + nGhostNodes();
    }

    KOKKOS_INLINE_FUNCTION Index nGlobalNodes() const
    {
        assert(is_built_);
        return global_number_nodes_;
    }

    KOKKOS_INLINE_FUNCTION Index globalRowOffset() const
    {
        assert(is_built_);
        return global_row_offset_;
    }

    KOKKOS_INLINE_FUNCTION Index nRows() const
    {
        return nOwnedNodes();
    }

    KOKKOS_INLINE_FUNCTION Index nGlobalRows() const
    {
        return nGlobalNodes();
    }

    KOKKOS_INLINE_FUNCTION Index nIndices() const
    {
        KOKKOS_IF_ON_DEVICE(return row_ptr_(row_ptr_.extent(0) - 1);)
        KOKKOS_IF_ON_HOST(return row_ptr_h_(row_ptr_h_.extent(0) - 1);)
    }

    KOKKOS_INLINE_FUNCTION unsigned long long nGlobalIndices() const
    {
        assert(is_built_);
        return global_number_indices_;
    }

    KOKKOS_INLINE_FUNCTION const RowPtrView& offsets() const
    {
        assert(is_built_);
        assert(row_ptr_.size() > 1);
        return row_ptr_;
    }

    inline const RowPtrViewHost& offsetsHost() const
    {
        assert(is_built_);
        assert(row_ptr_h_.size() > 1);
        return row_ptr_h_;
    }

    KOKKOS_INLINE_FUNCTION const IndexView& indices() const
    {
        assert(is_built_);
        assert(static_cast<Index>(primary_indices_.size()) == this->nIndices());
        return primary_indices_;
    }

    inline const IndexViewHost& indicesHost() const
    {
        assert(is_built_);
        assert(static_cast<Index>(primary_indices_h_.size()) ==
               this->nIndices());
        return primary_indices_h_;
    }

    KOKKOS_INLINE_FUNCTION const IndexView& diagonalIndicesOffset() const
    {
        assert(is_built_);
        assert(static_cast<Index>(diagonal_row_offset_.size()) ==
               this->nRows());
        return diagonal_row_offset_;
    }

    inline const IndexViewHost& diagonalIndicesOffsetHost() const
    {
        assert(is_built_);
        assert(static_cast<Index>(diagonal_row_offset_h_.size()) ==
               this->nRows());
        return diagonal_row_offset_h_;
    }

    inline const std::vector<PackInfo>& getPackInfos() const
    {
        assert(is_built_);
        return pack_infos_;
    }

    KOKKOS_INLINE_FUNCTION GraphLayout getLayout() const
    {
        return layout_;
    }

    KOKKOS_INLINE_FUNCTION bool isLocalColumnOrder() const
    {
        return (layout_ & GraphLayout::ColumnIndexOrder__Local);
    }

    KOKKOS_INLINE_FUNCTION bool isGlobalColumnOrder() const
    {
        return (layout_ & GraphLayout::ColumnIndexOrder__Global);
    }

    KOKKOS_INLINE_FUNCTION bool isBuilt() const
    {
        return is_built_;
    }

    KOKKOS_INLINE_FUNCTION Index nnzOwned(const Index i_row) const
    {
        KOKKOS_IF_ON_DEVICE((assert(is_built_); assert(0 <= i_row);
                             assert(i_row < this->nRows());
                             return row_nnz_owned_[i_row];))
        KOKKOS_IF_ON_HOST((assert(is_built_); assert(0 <= i_row);
                           assert(i_row < this->nRows());
                           return row_nnz_owned_h_[i_row];))
    }

    KOKKOS_INLINE_FUNCTION const IndexView& nnzOwned() const
    {
        assert(is_built_);
        return row_nnz_owned_;
    }

    inline const IndexViewHost& nnzOwnedHost() const
    {
        assert(is_built_);
        return row_nnz_owned_h_;
    }

    KOKKOS_INLINE_FUNCTION Index nnzGhost(const Index i_row) const
    {
        KOKKOS_IF_ON_DEVICE((assert(is_built_); assert(0 <= i_row);
                             assert(i_row < this->nRows());
                             return row_nnz_ghost_[i_row];))
        KOKKOS_IF_ON_HOST((assert(is_built_); assert(0 <= i_row);
                           assert(i_row < this->nRows());
                           return row_nnz_ghost_h_[i_row];))
    }

    KOKKOS_INLINE_FUNCTION const IndexView& nnzGhost() const
    {
        assert(is_built_);
        return row_nnz_ghost_;
    }

    inline const IndexViewHost& nnzGhostHost() const
    {
        assert(is_built_);
        return row_nnz_ghost_h_;
    }

    KOKKOS_INLINE_FUNCTION Index localToGlobalIndex(const Index j_local) const
    {
        // only defined if `j_local` is owned
        // clang-format off
        assert(0 <= j_local &&
               "CRSNodeGraph::localToGlobalIndex: `j_local` must be owned");
        assert(j_local <= this->nOwnedNodes() &&
               "CRSNodeGraph::localToGlobalIndex: `j_local` must be owned (inclusive)"); // inclusive
        // clang-format on
        return j_local + this->global_row_offset_;
    }

    KOKKOS_INLINE_FUNCTION Index globalToLocalIndex(const Index j_global) const
    {
        // only defined if `j_global` is owned
        // clang-format off
        assert(this->global_row_offset_ <= j_global &&
               "CRSNodeGraph::localToGlobalIndex: `j_global` must be owned");
        assert(j_global <= this->global_row_offset_ + this->nOwnedNodes() &&
               "CRSNodeGraph::localToGlobalIndex: `j_global` must be owned (inclusive)"); // inclusive
        // clang-format on
        return j_global - this->global_row_offset_;
    }

    KOKKOS_FUNCTION const IndexView& localIndices() const;
    KOKKOS_FUNCTION const IndexView& globalIndices() const;

    KOKKOS_FUNCTION IndexSubview rowLocalIndices(const Index i_row) const;
    KOKKOS_FUNCTION IndexSubview rowGlobalIndices(const Index i_row) const;

    const IndexViewHost& localIndicesHost() const;
    const IndexViewHost& globalIndicesHost() const;

    IndexSubviewHost rowLocalIndicesHost(const Index i_row) const;
    IndexSubviewHost rowGlobalIndicesHost(const Index i_row) const;

    void getMemoryFootprint(MemoryFootprint& data) const;
    void serialize(std::ofstream& out) const;
    void deserialize(std::ifstream& in);

protected:
    MPI_Comm comm_;
    int rank_;
    int size_;

    Index n_owned_nodes_; // locally owned nodes
    Index n_ghost_nodes_; // nodes in graph not locally owned by process
    bool determine_n_ghosts_;
    bool is_built_;

    // CRS data structures
    RowPtrView row_ptr_{"row_ptr", 0}; // row offsets
    IndexView primary_indices_{"primary_indices",
                               0}; // main column index order (sorted)
    IndexView secondary_indices_{"secondary_indices",
                                 0}; // complement column index order
    // host mirrors
    RowPtrViewHost row_ptr_h_;
    IndexViewHost primary_indices_h_;
    IndexViewHost secondary_indices_h_;

    // per row nnz
    IndexView row_nnz_owned_{"row_nnz_owned", 0}; // nnz per row (owned)
    IndexView row_nnz_ghost_{"row_nnz_ghost", 0}; // nnz per row (total ghosts)
    IndexViewHost row_nnz_owned_h_;
    IndexViewHost row_nnz_ghost_h_;

    // diagonal
    IndexView diagonal_row_offset_{"diagonal_row_offset",
                                   0}; // index into *_indices_
    IndexViewHost diagonal_row_offset_h_;

    // MPI
    Index global_row_offset_;
    Index global_number_nodes_;
    unsigned long long global_number_indices_;
    std::vector<PackInfo> pack_infos_;

    // DAVEKOKKOS: removed pure virtual
    virtual void buildGraph_() {};
    virtual void computePackInfos_();
    virtual void computeDiagonalIndices_();

    void computeAuxiliaryData_();

    void computeGlobalRowOffset_();
    void computeGlobalGraphSize_();
    void computeRowNonZeros_();

    void sortPrimaryIndices_();

    static inline void permuteInPlace_(Index* indices,
                                       const std::vector<Index>& permute)
    {
        std::vector<bool> done(permute.size());
        for (Index i = 0; i < static_cast<Index>(permute.size()); ++i)
        {
            if (done[i])
            {
                continue;
            }
            done[i] = true;
            Index prev_j = i;
            Index j = permute[i];
            while (i != j)
            {
                std::swap(indices[prev_j], indices[j]);
                done[j] = true;
                prev_j = j;
                j = permute[j];
            }
        }
    }

    static inline void permuteCopy_(Index* indices,
                                    std::vector<Index>& buffer,
                                    const std::vector<Index>& permute)
    {
        buffer.resize(permute.size());
        for (Index i = 0; i < static_cast<Index>(permute.size()); i++)
        {
            buffer[i] = indices[permute[i]];
        }
        std::copy(buffer.begin(), buffer.end(), indices);
    }

    void setup_();
    void resetGraph_();

private:
    const GraphLayout layout_;

    Index filterGhostsForOwnerRank_(
        int& owner_rank,
        std::vector<Index>::const_iterator ghost_start, // must be sorted
        const std::vector<Index>::const_iterator ghost_end,
        const int myrank, // of caller
        const int size,
        const std::vector<Index>& n_local_nodes) const;
};

} /* namespace linearSolver */

#endif /* CRSNODEGRAPH_H_UI1807EK */
