#ifndef KOKKOSREDUCTIONUTILS_H
#define KOKKOSREDUCTIONUTILS_H

#include <Kokkos_Core.hpp>

// example usage for minimum
/*
constexpr int NumComponents = N * M;

using reduce_value_type = NgpUtils::Reduction::FixedArray<scalar,
NumComponents>;

using reducer_type = NgpUtils::Reduction::FixedArrayMin<scalar, NumComponents>;

reduce_value_type result;
reducer_type reducer(result);

NgpUtils::Common::for_each_edge_side_reduce(
    ngpMesh,
    selector,
    geo,
    fields,
    reducer,
    workItem);

Kokkos::fence();

for (int c = 0; c < NumComponents; ++c) {
    scalar min_c = result[c];
}

// example usage for sum
same as above, except:
using reducer_type = NgpUtils::Reduction::FixedArraySum<scalar, NumComponents>;
*/

namespace linearSolver
{
namespace NgpUtils
{
namespace Reduction
{

// -----------------------------------------------------------------------------
// Fixed-size array reduction value
// -----------------------------------------------------------------------------

template <class Scalar, int N>
struct FixedArray
{
    static_assert(N > 0, "FixedArray reduction size N must be positive.");

    Scalar v[N];

    KOKKOS_INLINE_FUNCTION
    FixedArray() = default;

    KOKKOS_INLINE_FUNCTION
    Scalar& operator[](const int i)
    {
        return v[i];
    }

    KOKKOS_INLINE_FUNCTION
    const Scalar& operator[](const int i) const
    {
        return v[i];
    }

    KOKKOS_INLINE_FUNCTION
    void fill(const Scalar value)
    {
        for (int i = 0; i < N; ++i)
        {
            v[i] = value;
        }
    }
};

// -----------------------------------------------------------------------------
// Componentwise reduction operations
// -----------------------------------------------------------------------------

namespace ops
{

struct Sum
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::sum();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        dest += src;
    }
};

struct Prod
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::prod();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        dest *= src;
    }
};

struct Min
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::min();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        if (src < dest)
        {
            dest = src;
        }
    }
};

struct Max
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::max();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        if (src > dest)
        {
            dest = src;
        }
    }
};

struct LAnd
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::land();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        dest = dest && src;
    }
};

struct LOr
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::lor();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        dest = dest || src;
    }
};

struct BAnd
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::band();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        dest = dest & src;
    }
};

struct BOr
{
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar identity()
    {
        return Kokkos::reduction_identity<Scalar>::bor();
    }

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void join(Scalar& dest, const Scalar& src)
    {
        dest = dest | src;
    }
};

} // namespace ops

// -----------------------------------------------------------------------------
// Generic componentwise fixed-array reducer
// -----------------------------------------------------------------------------

template <class Op, class Scalar, int N, class Space = Kokkos::HostSpace>
struct FixedArrayReducer
{
public:
    using reducer = FixedArrayReducer;
    using value_type = FixedArray<Scalar, N>;

    using result_view_type =
        Kokkos::View<value_type*, Space, Kokkos::MemoryUnmanaged>;

private:
    value_type& value_;

public:
    KOKKOS_INLINE_FUNCTION
    explicit FixedArrayReducer(value_type& value) : value_(value)
    {
    }

    KOKKOS_INLINE_FUNCTION
    explicit FixedArrayReducer(const result_view_type& value) : value_(value(0))
    {
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& value) const
    {
        for (int i = 0; i < N; ++i)
        {
            value[i] = Op::template identity<Scalar>();
        }
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const
    {
        for (int i = 0; i < N; ++i)
        {
            Op::join(dest[i], src[i]);
        }
    }

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const
    {
        return value_;
    }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const
    {
        return result_view_type(&value_, 1);
    }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const
    {
        return true;
    }
};

// -----------------------------------------------------------------------------
// Convenient aliases
// -----------------------------------------------------------------------------

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArraySum = FixedArrayReducer<ops::Sum, Scalar, N, Space>;

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArrayProd = FixedArrayReducer<ops::Prod, Scalar, N, Space>;

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArrayMin = FixedArrayReducer<ops::Min, Scalar, N, Space>;

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArrayMax = FixedArrayReducer<ops::Max, Scalar, N, Space>;

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArrayLAnd = FixedArrayReducer<ops::LAnd, Scalar, N, Space>;

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArrayLOr = FixedArrayReducer<ops::LOr, Scalar, N, Space>;

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArrayBAnd = FixedArrayReducer<ops::BAnd, Scalar, N, Space>;

template <class Scalar, int N, class Space = Kokkos::HostSpace>
using FixedArrayBOr = FixedArrayReducer<ops::BOr, Scalar, N, Space>;

} // namespace Reduction
} // namespace NgpUtils
} // namespace linearSolver
#endif // KOKKOSREDUCTIONUTILS_H
