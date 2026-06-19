#ifndef LINEARSOLVERTYPES_H
#define LINEARSOLVERTYPES_H

#ifndef NO_LINSOLVE_CONF
#include "Config.h"
#endif /* NO_LINSOLVE_CONF */

#include <cstdint>

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

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
// #include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_BsrMatrix.hpp>

using AccelExecSpace = Kokkos::DefaultExecutionSpace;
using AccelMemorySpace = AccelExecSpace::memory_space;
using AccelDeviceType = Kokkos::Device<AccelExecSpace, AccelMemorySpace>;

// using CRSMatrixType = KokkosSparse::
//     CrsMatrix<TRealSolver, TGraphIndex, AccelDeviceType, void, TGraphIndex>;
using CRSMatrixType = KokkosSparse::Experimental::
    BsrMatrix<TRealSolver, TGraphIndex, AccelDeviceType, void, TGraphIndex>;

using CRSGraphType = typename CRSMatrixType::staticcrsgraph_type;
using RowPtrView = typename CRSGraphType::row_map_type;
using IndexView = typename CRSGraphType::entries_type;
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

NOT_SUPPORTED

#endif // USE_KOKKOS

#endif // LINEARSOLVERTYPES_H
