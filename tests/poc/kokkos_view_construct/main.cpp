#include <Kokkos_Core.hpp>
#include <iostream>

template <class ExecSpace>
void testKernel(const size_t s)
{
    using View = Kokkos::View<int*, ExecSpace>;
    View a;
    View b{"array"};
    View c{"array", 0};

    std::cout << a.is_allocated() << std::endl;
    std::cout << b.is_allocated() << std::endl;
    std::cout << c.is_allocated() << std::endl;

    std::cout << a.extent(0) << std::endl;
    std::cout << b.extent(0) << std::endl;
    std::cout << c.extent(0) << std::endl;

    typename View::HostMirror h_a = Kokkos::create_mirror_view(a);
    typename View::HostMirror h_b = Kokkos::create_mirror_view(b);
    typename View::HostMirror h_c = Kokkos::create_mirror_view(c);

    std::cout << h_a.label() << std::endl;
    std::cout << h_b.label() << std::endl;
    std::cout << h_c.label() << std::endl;

    Kokkos::resize(a, s);
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        testKernel<Kokkos::OpenMP>(16);
    }
    Kokkos::finalize();
    return 0;
}
