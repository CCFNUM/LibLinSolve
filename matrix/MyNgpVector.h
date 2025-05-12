#ifndef MY_NGP_VECTOR_H
#define MY_NGP_VECTOR_H

#include "Kokkos_Core.hpp"
#include <Kokkos_DualView.hpp>

// #include <iostream>

namespace linearSolver
{

template <typename Datatype>
class MyNgpVector
{

public:
    MyNgpVector(const std::string& n) : MyNgpVector(n, 0)
    {
    }

    //TODO: not calling the other constructor to allow fully empty construction
    MyNgpVector() // : MyNgpVector(get_default_name())
    {
    }

    MyNgpVector(const std::string& n, size_t s) : mSize(s), values(n, s)
    {
    }

    MyNgpVector(size_t s) : MyNgpVector(get_default_name(), s)
    {
    }

    MyNgpVector(const std::string& n, size_t s, Datatype init)
        : MyNgpVector(n, s)
    {
        Kokkos::deep_copy(values.view_host(), init);
        values.modify_host();
        values.sync_device();
    }

    MyNgpVector(size_t s, Datatype init)
        : MyNgpVector(get_default_name(), s, init)
    {
    }

    /***  HOST ONLY FUNCTIONS ***/

    std::string name() const
    {
        // return values.view_host().label();
        return values.label();
    }

    void fill_host(const Datatype& val) const
    {
        Kokkos::deep_copy(values.view_host(), val);
    }

    void fill_device(const Datatype& val) const
    {
        Kokkos::deep_copy(values.view_device(), val);
    }

    void fill(const Datatype& val) const
    {
        fill_host(val);
        fill_device(val);
    }

    size_t capacity() const
    {
        return values.view_host().size();
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
                values.view_host()(i) = init;
        mSize = s;
        values.modify_host();
        values.sync_device();
    }

    void clear()
    {
        mSize = 0;
        values.resize(mSize);
        values.clear_sync_state();
    }

    void push_back(const Datatype& val)
    {
        if (mSize >= capacity())
            grow_to_size(mSize + get_push_back_increase_size());
        values.view_host()[mSize] = val;
        mSize++;
    }

    void manual_copy_host_to_device()
    {
        Kokkos::deep_copy(values.view_device(), values.view_host());
    }

    void manual_copy_device_to_host()
    {
        Kokkos::deep_copy(values.view_host(), values.view_device());
    }

    void modifyHost()
    {
        values.modify_host();
    }

    void modifyDevice()
    {
        values.modify_device();
    }

    void syncToHost()
    {
        values.sync_host();
    }

    void syncToDevice()
    {
        values.sync_device();
    }

    // debug
    bool isAllocated()
    {
        return values.is_allocated();
    }

    auto get_host_view()
    {
        return values.view_host();
    }

    /***  HOST/DEVICE FUNCTIONS ***/
    // TODO: don't like the size member. should be obtained from view directly
    KOKKOS_INLINE_FUNCTION size_t size() const
    {
        return mSize;
    }

    KOKKOS_INLINE_FUNCTION bool empty() const
    {
        return mSize == 0;
    }

    KOKKOS_INLINE_FUNCTION
    auto data() const
    {
        KOKKOS_IF_ON_DEVICE(return values.view_device().data();)
        KOKKOS_IF_ON_HOST(return values.view_host().data();)
    }

    // TODO: might need to be const too
    KOKKOS_INLINE_FUNCTION
    Datatype& back()
    {
        return this->operator[](this->size() - 1);
    }

    KOKKOS_INLINE_FUNCTION
    const Datatype& back() const
    {
        return this->operator[](this->size() - 1);
    }

    KOKKOS_INLINE_FUNCTION
    Datatype& operator[](size_t i) const
    {
        KOKKOS_IF_ON_DEVICE(return values.view_device()(i);)
        KOKKOS_IF_ON_HOST(return values.view_host()(i);)
    }

    // debug
    KOKKOS_INLINE_FUNCTION
    auto get_device_view() const
    {
        return values.view_device();
    }

protected:
    // #ifdef KOKKOS_ENABLE_CUDA
    //     using DeviceSpace = Kokkos::CudaSpace;
    // #elif defined(KOKKOS_ENABLE_HIP)
    //     using DeviceSpace = Kokkos::HIPSpace;
    // #else
    //     using DeviceSpace = Kokkos::HostSpace;
    // #endif
    using DualType = Kokkos::DualView<Datatype*>;
    using DeviceType = typename DualType::t_dev;
    using HostType = typename DualType::t_host;
    using DeviceSpace = DeviceType::memory_space;
    using HostSpace = HostType::memory_space;

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
            return Kokkos::subview(values.view_device(),
                                   Kokkos::make_pair(begin, end));)

        KOKKOS_IF_ON_HOST(
            return Kokkos::subview(values.view_host(),
                                   Kokkos::make_pair(begin, end));)
    }

    KOKKOS_FUNCTION
    auto subview_const(size_t begin, size_t end) const
    {
        KOKKOS_IF_ON_DEVICE(return ConstSubviewTypeDevice(Kokkos::subview(
            values.view_device(), Kokkos::make_pair(begin, end)));)
        KOKKOS_IF_ON_HOST(return ConstSubviewTypeHost(Kokkos::subview(
            values.view_host(), Kokkos::make_pair(begin, end)));)
    }

    KOKKOS_FUNCTION
    auto subview_all() const
    {
        KOKKOS_IF_ON_DEVICE(
            return Kokkos::subview(values.view_device(), Kokkos::ALL);)

        KOKKOS_IF_ON_HOST(
            return Kokkos::subview(values.view_host(), Kokkos::ALL);)
        // auto tmp = Kokkos::subview(values, Kokkos::make_pair(size_t(0),
        // mSize)); KOKKOS_IF_ON_DEVICE(return tmp.view_device();)
        // KOKKOS_IF_ON_HOST(return tmp.view_host();)
    }

    KOKKOS_FUNCTION
    auto subview_all_const() const
    {
        KOKKOS_IF_ON_DEVICE(return ConstSubviewTypeDevice(Kokkos::subview(
            values.view_device(), Kokkos::ALL));)
        KOKKOS_IF_ON_HOST(return ConstSubviewTypeHost(Kokkos::subview(
            values.view_host(), Kokkos::ALL));)
        // return subview_all();
    }

private:
    size_t mSize = 0;
    DualType values;

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
        // this resizes and copies existing data into the new size.
        // this happens in the space where the device was last modified.
        // at the start, this modifies device!
        values.resize(s);
        if (values.need_sync_host())
        {
            values.sync_host();
        }
        else if (values.need_sync_device())
        {
            values.sync_device();
        }
    }
};

} /* namespace linearSolver */

#endif /* NGP_VECTOR_H */
