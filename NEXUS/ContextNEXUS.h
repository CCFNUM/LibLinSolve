// File       : ContextNEXUS.h
// Description: Solver context for the NEXUS linear solver library
//              (https://github.com/CCFNUM/NEXUS).  NEXUS stores none of the
//              components of the linear system: this context attaches the
//              framework's coefficient storage (LHS values, solution, RHS)
//              and the LU factorization buffers of coefficients::LUData to
//              NEXUS zero-copy through its stable C API.  On device-resident
//              builds (CUDA/HIP Kokkos) host mirrors are attached instead
//              and synchronized around each solve, since NEXUS consumes host
//              memory.
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef CONTEXTNEXUS_H_LINSOLVE
#define CONTEXTNEXUS_H_LINSOLVE

#ifndef NO_LINSOLVE_CONF
#include "Config.h"
#endif /* NO_LINSOLVE_CONF */
#include "linearSolverContext.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#ifdef HAS_NEXUS
#include <nexus/nexus.h>
#endif /* HAS_NEXUS */

namespace linearSolver
{

template <size_t N>
class ContextNEXUS : public Context<N>
{
public:
    using Base = Context<N>;
    using Coefficients = typename Base::Coefficients;
    using Vector = typename Base::Vector;
    using Index = typename Base::Index;
    using Base::BLOCKSIZE;

    ContextNEXUS() = delete;

    ContextNEXUS(const std::string& family,
                 const std::string& system_name,
                 const CRSNodeGraph* graph,
                 const YAML::Node* node = nullptr)
        : Base(ID::AMG, family, system_name, graph, node)
    {
        initializeNEXUS_();
    }

    ContextNEXUS(const std::string& family,
                 const std::string& system_name,
                 std::shared_ptr<Coefficients> coeffs,
                 const YAML::Node* node = nullptr)
        : Base(ID::AMG, family, system_name, coeffs, node)
    {
        initializeNEXUS_();
    }

    ~ContextNEXUS() override
    {
#ifdef HAS_NEXUS
        NEXUS_MatrixDestroy(&matrix_);
#endif /* HAS_NEXUS */
    }

#ifdef HAS_NEXUS
    NEXUS_Matrix getMatrixNEXUS()
    {
        return matrix_;
    }

    // push the assembled system to the storage NEXUS sees (no-op when the
    // coefficient views are host-accessible: NEXUS reads them in place)
    void uploadSystem()
    {
        if (!host_accessible_())
        {
            Kokkos::deep_copy(values_mirror_, this->coeffs()->valuesRef());
            Kokkos::deep_copy(x_mirror_, this->coeffs()->getXVector());
            Kokkos::deep_copy(b_mirror_, this->coeffs()->getBVector());
        }
    }

    // pull the solution back into the coefficient views (no-op when NEXUS
    // wrote into them directly)
    void downloadSolution()
    {
        if (!host_accessible_())
        {
            Kokkos::deep_copy(this->coeffs()->getXVector(), x_mirror_);
            // The LU factorization results remain in the attached host
            // buffers; the device-side LUData views are not refreshed (the
            // framework does not consume them, they exist as driver-owned
            // storage for NEXUS).
        }
    }
#endif /* HAS_NEXUS */

private:
#ifdef HAS_NEXUS
    NEXUS_Matrix matrix_ = nullptr;

    // host mirrors for device-resident coefficient storage
    typename Vector::host_mirror_type values_mirror_;
    typename Vector::host_mirror_type x_mirror_;
    typename Vector::host_mirror_type b_mirror_;
    typename Vector::host_mirror_type lu_mirror_;
    typename Vector::host_mirror_type d_mirror_;

    static constexpr bool host_accessible_()
    {
        return Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                          AccelMemorySpace>::accessible;
    }

    void initializeNEXUS_()
    {
        auto coeffs = this->coeffs();
        const CRSNodeGraph& graph = coeffs->getGraph();

        int comm_size = 1;
        MPI_Comm_size(graph.getCommunicator(), &comm_size);
        if (comm_size > 1)
        {
            throw std::runtime_error(
                "ContextNEXUS: multi-rank systems are not supported yet "
                "(column ordering permutation pending); run this system on "
                "a single rank or use another solver family");
        }

        // CRS pattern from the graph host arrays (single rank: the global
        // column ordering coincides with the local one)
        const auto offsets = graph.offsetsHost();
        const auto indices = graph.indicesHost();
        const Index n_rows = graph.nRows();
        std::vector<int> row_ptr(n_rows + 1);
        for (Index i = 0; i <= n_rows; ++i)
        {
            row_ptr[i] = static_cast<int>(offsets(i));
        }
        std::vector<int> col_idx(graph.nIndices());
        for (Index k = 0; k < graph.nIndices(); ++k)
        {
            col_idx[k] = static_cast<int>(indices(k));
        }

        callNEXUS_(NEXUS_MatrixCreate(graph.getCommunicator(),
                                      static_cast<int>(N),
                                      static_cast<int>(n_rows),
                                      row_ptr.data(),
                                      col_idx.data(),
                                      &matrix_),
                   "NEXUS_MatrixCreate");

        // The framework owns every component of the linear system,
        // including the LU factorization buffers of coefficients::LUData
        // (sized here once; filled by the NEXUS smoother setup).
        auto& lu = coeffs->getLUData();
        const size_t nnz = static_cast<size_t>(coeffs->nnz());
        const size_t ndiag = static_cast<size_t>(N * N) * n_rows;
        lu.U_offsets.resize(static_cast<size_t>(n_rows), Index(-1));
        Kokkos::resize(lu.LU_values, nnz);
        Kokkos::resize(lu.D_values, ndiag);

        double* values_ptr = nullptr;
        double* x_ptr = nullptr;
        double* b_ptr = nullptr;
        double* lu_ptr = nullptr;
        double* d_ptr = nullptr;
        if (host_accessible_())
        {
            // zero-copy: NEXUS works directly on the framework storage
            values_ptr = coeffs->valuesRef().data();
            x_ptr = coeffs->getXVector().data();
            b_ptr = coeffs->getBVector().data();
            lu_ptr = lu.LU_values.data();
            d_ptr = lu.D_values.data();
        }
        else
        {
            // device-resident storage: attach host mirrors and synchronize
            // around the solves (uploadSystem/downloadSolution)
            values_mirror_ =
                Kokkos::create_mirror_view(coeffs->valuesRef());
            x_mirror_ = Kokkos::create_mirror_view(coeffs->getXVector());
            b_mirror_ = Kokkos::create_mirror_view(coeffs->getBVector());
            lu_mirror_ = Kokkos::create_mirror_view(lu.LU_values);
            d_mirror_ = Kokkos::create_mirror_view(lu.D_values);
            values_ptr = values_mirror_.data();
            x_ptr = x_mirror_.data();
            b_ptr = b_mirror_.data();
            lu_ptr = lu_mirror_.data();
            d_ptr = d_mirror_.data();
        }

        callNEXUS_(NEXUS_MatrixAttachSystem(matrix_, values_ptr, x_ptr,
                                            b_ptr),
                   "NEXUS_MatrixAttachSystem");
        callNEXUS_(NEXUS_MatrixAttachLUData(matrix_,
                                            lu.U_offsets.data(),
                                            lu_ptr,
                                            d_ptr),
                   "NEXUS_MatrixAttachLUData");
    }

    static void callNEXUS_(const int status, const char* what)
    {
        if (status != 0)
        {
            throw std::runtime_error(std::string(what) + ": " +
                                     NEXUS_ErrorMessage());
        }
    }
#else
    void initializeNEXUS_()
    {
        throw std::runtime_error(
            "ContextNEXUS: not supported by this build (configure "
            "libLinSolve with NEXUS support and define HAS_NEXUS)");
    }
#endif /* HAS_NEXUS */
};

} /* namespace linearSolver */

#endif /* CONTEXTNEXUS_H_LINSOLVE */
