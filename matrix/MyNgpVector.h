#ifndef NGP_VECTOR_H
#define NGP_VECTOR_H

#include "Kokkos_Core.hpp"

namespace linearSolver
{

template <typename Datatype>
class MyNgpVector
{
    using HostSpace = Kokkos::DefaultHostExecutionSpace;

public:
    MyNgpVector(const std::string& n) : MyNgpVector(n, 0)
    {
    }

    MyNgpVector() : MyNgpVector(get_default_name())
    {
    }

    MyNgpVector(const std::string& n, size_t s)
        : mSize(s),
          deviceVals(Kokkos::view_alloc(Kokkos::WithoutInitializing, n), mSize),
          hostVals(Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                              HostSpace(),
                                              deviceVals))
    {
    }

    MyNgpVector(size_t s) : MyNgpVector(get_default_name(), s)
    {
    }

    MyNgpVector(const std::string& n, size_t s, Datatype init)
        : MyNgpVector(n, s)
    {
        Kokkos::deep_copy(hostVals, init);
    }

    MyNgpVector(size_t s, Datatype init)
        : MyNgpVector(get_default_name(), s, init)
    {
    }

    std::string name() const
    {
        return hostVals.label();
    }

    auto view_host()
    {
        return Kokkos::subview(hostVals, Kokkos::make_pair(size_t(0), size()));
    }

    KOKKOS_INLINE_FUNCTION
    auto view_device()
    {
        return Kokkos::subview(deviceVals,
                               Kokkos::make_pair(size_t(0), size()));
    }

    auto subview_host(size_t begin, size_t end) const
    {
        return Kokkos::subview(hostVals, Kokkos::make_pair(begin, end));
    }

    KOKKOS_INLINE_FUNCTION
    auto subview_device(size_t begin, size_t end) const
    {
        return Kokkos::subview(deviceVals, Kokkos::make_pair(begin, end));
    }

    // auto data_host() const
    // {
    //     return hostVals.data();
    // }

    KOKKOS_INLINE_FUNCTION
    auto data() const
    {
        KOKKOS_IF_ON_DEVICE(return deviceVals.data();)
        KOKKOS_IF_ON_HOST(return hostVals.data();)
    }

    // KOKKOS_INLINE_FUNCTION
    // auto data_device() const
    // {
    //     return deviceVals.data();
    // }

    KOKKOS_FUNCTION
    Datatype& back()
    {
        return this->operator[](this->size() - 1);
    }

    KOKKOS_FUNCTION
    const Datatype& back() const
    {
        return this->operator[](this->size() - 1);
    }

    void fill_host(const Datatype& val)
    {
        Kokkos::deep_copy(hostVals, val);
    }

    KOKKOS_INLINE_FUNCTION
    void fill_device(const Datatype& val)
    {
        Kokkos::deep_copy(deviceVals, val);
    }

    KOKKOS_FUNCTION size_t size() const
    {
        return mSize;
    }

    KOKKOS_FUNCTION bool empty() const
    {
        return mSize == 0;
    }

    size_t capacity() const
    {
        return hostVals.size();
    }

    void resize(size_t s)
    {
        resize(s, 0);
    }

    void resize(size_t s, Datatype init)
    {
        if (s > capacity())
            grow_to_size(s);
        if (s > mSize)
            for (size_t i = mSize; i < s; i++)
                hostVals(i) = init;
        mSize = s;
    }

    void clear()
    {
        mSize = 0;
    }

    KOKKOS_INLINE_FUNCTION
    Datatype& operator[](size_t i) const
    {
        KOKKOS_IF_ON_DEVICE(return deviceVals(i);)
        KOKKOS_IF_ON_HOST(return hostVals(i);)
    }

    // KOKKOS_FUNCTION Datatype& device_get(size_t i) const
    // {
    //     return deviceVals(i);
    // }

protected:
#ifdef KOKKOS_ENABLE_CUDA
    using DeviceSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
    using DeviceSpace = Kokkos::HIPSpace;
#else
    using DeviceSpace = Kokkos::HostSpace;
#endif
public:
    template <class Device>
    KOKKOS_FUNCTION Datatype&
    get(typename std::enable_if<
        std::is_same<typename Device::execution_space,
                     DeviceSpace::execution_space>::value,
        size_t>::type i) const
    {
        return deviceVals(i);
    }
#ifdef STK_ENABLE_GPU
    template <class Device>
    KOKKOS_FUNCTION Datatype&
    get(typename std::enable_if<
        !std::is_same<typename Device::execution_space,
                      DeviceSpace::execution_space>::value,
        size_t>::type i) const
    {
        return hostVals(i);
    }
#endif

    void push_back(Datatype val)
    {
        if (mSize >= capacity())
            grow_to_size(mSize + get_push_back_increase_size());
        hostVals[mSize] = val;
        mSize++;
    }

    void copy_host_to_device()
    {
        Kokkos::deep_copy(deviceVals, hostVals);
    }

    void copy_device_to_host()
    {
        Kokkos::deep_copy(hostVals, deviceVals);
    }

protected:
    using DeviceType = Kokkos::View<Datatype*, DeviceSpace>;
    using HostType = typename DeviceType::HostMirror;

    virtual DeviceType get_new_vals_of_size(size_t s)
    {
        return DeviceType(deviceVals.label(), s);
    }

public:
    using SubviewTypeHost =
        typename Kokkos::Subview<HostType, Kokkos::pair<unsigned, unsigned>>;
    using ConstSubviewTypeHost =
        typename Kokkos::Subview<Kokkos::View<const Datatype*, HostSpace>,
                                 Kokkos::pair<unsigned, unsigned>>;
    using SubviewTypeDevice =
        typename Kokkos::Subview<DeviceType, Kokkos::pair<unsigned, unsigned>>;
    using ConstSubviewTypeDevice =
        typename Kokkos::Subview<Kokkos::View<const Datatype*, DeviceSpace>,
                                 Kokkos::pair<unsigned, unsigned>>;

    KOKKOS_FUNCTION
    auto subview(size_t begin, size_t end) const
    {
        KOKKOS_IF_ON_DEVICE(
            return Kokkos::subview(deviceVals, Kokkos::make_pair(begin, end));)

        KOKKOS_IF_ON_HOST(
            return Kokkos::subview(hostVals, Kokkos::make_pair(begin, end));)
    }

    KOKKOS_FUNCTION
    auto subview_const(size_t begin, size_t end) const
    {
        KOKKOS_IF_ON_DEVICE(return ConstSubviewTypeDevice(Kokkos::subview(
            deviceVals, Kokkos::make_pair(begin, end)));)
        KOKKOS_IF_ON_HOST(return ConstSubviewTypeHost(Kokkos::subview(
            hostVals, Kokkos::make_pair(begin, end)));)
    }

    KOKKOS_FUNCTION
    auto subview_all() const
    {
        KOKKOS_IF_ON_DEVICE(return Kokkos::subview(deviceVals, Kokkos::ALL);)

        KOKKOS_IF_ON_HOST(return Kokkos::subview(hostVals, Kokkos::ALL);)
    }

    KOKKOS_FUNCTION
    auto subview_all_const() const
    {
        KOKKOS_IF_ON_DEVICE(return ConstSubviewTypeDevice(Kokkos::subview(
            deviceVals, Kokkos::ALL));)
        KOKKOS_IF_ON_HOST(return ConstSubviewTypeHost(
                                     Kokkos::subview(hostVals, Kokkos::ALL));)
    }

private:
    size_t mSize;
    DeviceType deviceVals;
    HostType hostVals;

    static const char* get_default_name()
    {
        return "UnnamedStkVector";
    }

    size_t get_push_back_increase_size() const
    {
        if (mSize == 0)
            return 1;
        return mSize;
    }

    void grow_to_size(size_t s)
    {
        deviceVals = get_new_vals_of_size(s);
        HostType tmp = Kokkos::create_mirror_view(deviceVals);
        copy_into_bigger(tmp, hostVals);
        hostVals = tmp;
    }

    void copy_into_bigger(HostType& dst, HostType& src)
    {
        for (size_t i = 0; i < src.size(); i++)
            dst[i] = src[i];
    }
};

} /* namespace linearSolver */

#endif /* NGP_VECTOR_H */