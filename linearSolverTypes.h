#ifndef LINEARSOLVERTYPES_H
#define LINEARSOLVERTYPES_H

#ifndef NO_LINSOLVE_CONF
#include "Config.h"
#endif /* NO_LINSOLVE_CONF */

#include <cstdint>

// FIXME [faw 2026-06-23]: this is the column index type under Kokkos!  With the
// RowIndex and ColINdex typedefs below the TGraphIndex should not be used
// anywhere except possibly in this file for readability.  It can be #undef'ed
// at the end.
#ifdef GRAPH_INDEX_64BIT
typedef int64_t TGraphIndex;
#else
typedef int32_t TGraphIndex;
#endif /* GRAPH_INDEX_64BIT */

#ifdef SOLVER_SINGLE_PRECISION
typedef float TRealSolver;
#else
typedef double TRealSolver;
#endif /* SOLVER_SINGLE_PRECISION */

#ifdef HAS_KOKKOS
#include <Kokkos_Core.hpp>
// #include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_BsrMatrix.hpp>

using AccelExecSpace = Kokkos::DefaultExecutionSpace;
using AccelMemorySpace = AccelExecSpace::memory_space;
using AccelDeviceType = Kokkos::Device<AccelExecSpace, AccelMemorySpace>;

using CRSRowMatrixType = KokkosSparse::
    CrsMatrix<TRealSolver, TGraphIndex, AccelDeviceType, void, size_t>;
using CRSBlockMatrixType = KokkosSparse::Experimental::
    BsrMatrix<TRealSolver, TGraphIndex, AccelDeviceType, void, size_t>;

using CRSMatrixType = CRSBlockMatrixType;

using CRSGraphType = typename CRSMatrixType::staticcrsgraph_type;
using RowIndex = typename CRSGraphType::size_type;
using ColIndex = typename CRSGraphType::data_type; // used for Index type alias
using RowPtrView = typename CRSGraphType::row_map_type::non_const_type;
using IndexView = typename CRSGraphType::entries_type;
using RowPtrViewHost = typename RowPtrView::HostMirror;
using IndexViewHost = typename IndexView::HostMirror;

using ScalarView = typename CRSMatrixType::values_type;

using RangeType = Kokkos::pair<TGraphIndex, TGraphIndex>;
using RowPtrSubview = Kokkos::Subview<RowPtrView, RangeType>;
using IndexSubview = Kokkos::Subview<IndexView, RangeType>;
using IndexSubviewHost = Kokkos::Subview<IndexViewHost, RangeType>;
using ScalarSubview = Kokkos::Subview<ScalarView, RangeType>;

// RowPtrSubview: view<const TGraphIndex*>
// IndexSubview: view<TGraphIndex*>
// ScalarSubview: view<DataType*>
#else

// nop

#endif // HAS_KOKKOS

#endif // LINEARSOLVERTYPES_H
