// File       : CRSMatrix.h
// Created    : Thu Mar 13 2025 10:58:05 (+0100)
// Author     : Fabian Wermelinger
// Description: Compressed row storage matrix
// Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef CRSMatrix_h
#define CRSMatrix_h

#include "CRSConnectivity.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <span>
#include <vector>

#ifdef SOLVER_SINGLE_PRECISION
typedef float TRealSolver;
#else
typedef double TRealSolver;
#endif /* SOLVER_SINGLE_PRECISION */

namespace linearSolver
{

template <size_t N>
class CRSMatrix : public CRSConnectivity
{
public:
    using MemoryLayout = GraphLayout;
    using Index = CRSConnectivity::Index;
    using DataType = TRealSolver;
    using Vector = std::vector<DataType>;

    static constexpr Index BLOCKSIZE = N;

    // DAVEKOKKOS: move accelexecspace to global
    using AccelExecSpace = Kokkos::DefaultExecutionSpace;
    using execution_space = AccelExecSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;

    // using matrix_type =
    //     KokkosSparse::CrsMatrix<DataType, Index, device_type, void, Index>;
    using matrix_type = KokkosSparse::Experimental::
        BsrMatrix<DataType, Index, device_type, void, Index>;

    using graph_type = typename matrix_type::staticcrsgraph_type;
    using row_map_type = typename graph_type::row_map_type;
    using entries_type = typename graph_type::entries_type;
    using values_type = typename matrix_type::values_type;

    using range_type = Kokkos::pair<Index, Index>;
    using row_map_subview_type = Kokkos::Subview<row_map_type, range_type>;
    using entries_subview_type = Kokkos::Subview<entries_type, range_type>;
    using values_subview_type = Kokkos::Subview<values_type, range_type>;

private:
    template <typename T>
    std::vector<char> gatherToGlobal_(const T* values,
                                      const size_t n_values,
                                      const bool exscan = false) const;
    void dumpBinary_(const std::string fname, const std::vector<char>& v) const;

    void writeMatrix_(const char* basename) const;

    std::ostream& stream_(std::ostream& os,
                          Index max_rows = 0,
                          Index max_cols = 0,
                          const Index width = 20,
                          const Index precision = 14) const;

    friend std::ostream& operator<<(std::ostream& os, const CRSMatrix& mat)
    {
        return mat.stream_(os);
    }

protected:
    MemoryLayout memory_layout_;
    values_type values_; // flat values array
    matrix_type crsmatrix_;

public:
    CRSMatrix() = delete;

    // Majority of code depends on a block-structured row-major memory layout.
    // Local conversion is carried out where necessary.  Changing the matrix
    // memory layout in constructors below will most certainly break the code.
    CRSMatrix(const CRSNodeGraph* graph, const bool allocate_values = true)
        : CRSConnectivity(graph),
          memory_layout_(graph->getLayout() |
                         MemoryLayout::MemoryLayout__BlockRowMajor)
    {
        if (allocate_values)
        {
            Kokkos::resize(values_, this->nnz());
            Kokkos::deep_copy(values_, static_cast<DataType>(0));
        }

        graph_type mygraph(graph->indices(), graph->offsets());
        crsmatrix_ = matrix_type(
            "CRSMatrix", this->nRows(), values_, mygraph, BLOCKSIZE);
    }

    // Access / on-the-fly operations
    inline Index nnz() const
    {
        return BLOCKSIZE * BLOCKSIZE * this->nnzBlocks();
    }

    inline unsigned long long nnzGlobal() const
    {
        return BLOCKSIZE * BLOCKSIZE * this->nnzGlobalBlocks();
    }

    inline DataType* valuesPtr()
    {
        return values_.data();
    }

    inline const DataType* valuesPtr() const
    {
        return values_.data();
    }

    inline values_type& valuesRef()
    {
        return values_;
    }

    inline const values_type& valuesRef() const
    {
        return values_;
    }

    inline values_subview_type rowVals(Index iRow) const
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        const auto& offsets = this->offsetsRef();
        return Kokkos::subview(
            values_,
            Kokkos::make_pair(BLOCKSIZE * BLOCKSIZE * offsets[iRow],
                              BLOCKSIZE * BLOCKSIZE * offsets[iRow + 1]));
    }

    inline auto blockRowVals(Index iRow) const
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        return crsmatrix_.block_row(iRow);
    }

    inline entries_subview_type blockRowCols(Index iRow) const
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        const Index begin = crsmatrix_.graph.row_map(iRow);
        const Index end = crsmatrix_.graph.row_map(iRow + 1);
        return Kokkos::subview(crsmatrix_.graph.entries,
                               Kokkos::pair<Index, Index>(begin, end));
    }

    // inline const std::span<const DataType> rowVals(Index iRow) const
    // {
    //     assert(0 <= iRow);
    //     assert(iRow < this->nRows());
    //     const auto& offsets = this->offsetsRef();
    //     return std::span<const DataType>(values_).subspan(
    //         BLOCKSIZE * BLOCKSIZE * offsets[iRow],
    //         BLOCKSIZE * BLOCKSIZE * (offsets[iRow + 1] - offsets[iRow]));
    // }

    inline DataType& dofDiag(Index iRow, Index dof = 0)
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        const auto& diag_offsets = this->diagOffsetRef();
        return this->rowVals(iRow)[BLOCKSIZE * BLOCKSIZE * diag_offsets[iRow] +
                                   (BLOCKSIZE + 1) * dof];
    };

    inline const DataType& dofDiag(Index iRow, Index dof = 0) const
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        const auto& diag_offsets = this->diagOffsetRef();
        return this->rowVals(iRow)[BLOCKSIZE * BLOCKSIZE * diag_offsets[iRow] +
                                   (BLOCKSIZE + 1) * dof];
    };

    inline DataType* diag(Index iRow)
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        const auto& offsets = this->offsetsRef();
        const auto& diag_offsets = this->diagOffsetRef();
        return &values_[BLOCKSIZE * BLOCKSIZE * offsets[iRow] +
                        BLOCKSIZE * BLOCKSIZE * diag_offsets[iRow]];
    };

    inline const DataType* diag(Index iRow) const
    {
        assert(0 <= iRow);
        assert(iRow < this->nRows());
        const auto& offsets = this->offsetsRef();
        const auto& diag_offsets = this->diagOffsetRef();
        return &values_[BLOCKSIZE * BLOCKSIZE * offsets[iRow] +
                        BLOCKSIZE * BLOCKSIZE * diag_offsets[iRow]];
    };

    // layout conversions
    MemoryLayout getMemoryLayout() const
    {
        return memory_layout_;
    }

    // Internal use only for debug purposes (expensive search !!): i and j are
    // absolute indices, not block indices
    DataType operator()(Index i, Index j) const;

    // IO

    void dump(Index maxRow = -1,
              Index maxCol = -1,
              Index width = 8,
              Index precision = 14) const;

    // this is slow because it uses operator() function
    void dumpRow(Index lid, Index width = 8, Index precision = 14) const;

    void writeMatrix(const std::string basename) const
    {
        this->writeMatrix_(basename.c_str());
    }

    // matrix properties
    Index bandwidth() const; // matrix maximum bandwidth

    Index profile() const; // matrix profile/envelope

    enum class MatrixNorm
    {
        Frobenius,
    };

    DataType norm(const MatrixNorm type = MatrixNorm::Frobenius) const;
};
} // namespace linearSolver

#include "CRSMatrix.hpp"

#endif // CRSMatrix_h
