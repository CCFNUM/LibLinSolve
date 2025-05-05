#include <Kokkos_Core.hpp>

#define N 8
typedef char T;

template <typename TView>
KOKKOS_INLINE_FUNCTION void printLayout(const TView v)
{
    const T* base = v.data();
    for (ptrdiff_t k = 0; k < N * N; k++)
    {
        const T* data = base + k;
        const int j = k / N;
        const int i = k % N;
        Kokkos::printf("(j=%d, i=%d):\tk=%d\t%p -> %d\t&view(j,i): %p -> %d\n",
                       j,
                       i,
                       k,
                       data,
                       *data,
                       &v(j, i),
                       v(j, i));
    }
}

template <class ExecSpace>
void testKernel(const char* what_space, const bool print_host = false)
{
    using My2DView = Kokkos::View<T[N][N], ExecSpace>;
    My2DView b("array");
    typename My2DView::HostMirror h_b = Kokkos::create_mirror_view(b);

    // host initialization
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            h_b(j, i) = j * N + i;
        }
    }
    Kokkos::deep_copy(b, h_b);

    // verbosity
    Kokkos::printf("\n%s\n", what_space);
    Kokkos::printf("Device ID: %d\n", Kokkos::device_id());
    Kokkos::printf("NThreads:  %d\n", Kokkos::num_threads());

    using host_layout = typename decltype(h_b)::layout_type;
    using device_layout = typename decltype(b)::layout_type;

    // view meta data
    // clang-format off
    Kokkos::printf("DEVICE VIEW:\n");
    Kokkos::printf("\tAddress:   %p\n", b.data());
    Kokkos::printf("\tExtent:    [%d, %d]\n", b.extent(0), b.extent(1));
    Kokkos::printf("\tStride:    [%d, %d]\n", b.stride_0(), b.stride_1());
    Kokkos::printf("\tLayout:    %s\n", typeid(device_layout).name());
    // Kokkos::printf("\tPadding:   %zu\n", device_layout::padding_value);
    Kokkos::printf("HOST VIEW:\n");
    Kokkos::printf("\tAddress:   %p\n", h_b.data());
    Kokkos::printf("\tExtent:    [%d, %d]\n", b.extent(0), b.extent(1));
    Kokkos::printf("\tStride:    [%d, %d]\n", b.stride_0(), b.stride_1());
    Kokkos::printf("\tLayout:    %s\n", typeid(host_layout).name());
    // Kokkos::printf("\tPadding:   %zu\n", host_layout::padding_value);
    // clang-format on

    if (print_host)
    {
        Kokkos::printf("Host base: %p\n", h_b.data());
        printLayout(h_b);
    }

    using range_policy = Kokkos::RangePolicy<ExecSpace>;
    Kokkos::parallel_for("iter", range_policy(0, 1), KOKKOS_LAMBDA(int dummy) {
        Kokkos::printf("Kernel base: %p\n", b.data());
        printLayout(b);
    });
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        testKernel<Kokkos::OpenMP>("Kokkos::OpenMP", true);
        testKernel<Kokkos::Cuda>("Kokkos::Cuda", true);
    }
    Kokkos::finalize();
    return 0;
}
