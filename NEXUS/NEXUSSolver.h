// File       : NEXUSSolver.h
// Description: NEXUS based linear solver wrapper (solver family "nexus").
//              NEXUS is a standalone linear solver library (AMG, GMRES,
//              DirectLU with host and Kokkos backends) driven, like the
//              PETSc/HYPRE wrappers, through its stable C API; the two
//              libraries share no C++ symbols.  NEXUS stores none of the
//              components of the linear system: the framework coefficient
//              storage (and its LU factorization buffers) is attached by
//              ContextNEXUS.
//
//              Configuration: the solver node must hold the NEXUS solver
//              configuration in a nested `nexus:` map, e.g.
//
//                  family: nexus
//                  nexus:
//                      family: GMRES
//                      options:
//                          krylov_type: flexgmres
//                          rtol: 1.0e-8
//                          precond:
//                              family: AMGSolver
//                              options: { cycle_type: v-cycle,
//                                         smoother_type: ilu0 }
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef NEXUSSOLVER_H_LINSOLVE
#define NEXUSSOLVER_H_LINSOLVE

#ifndef NO_LINSOLVE_CONF
#include "Config.h"
#endif /* NO_LINSOLVE_CONF */
#include "ContextNEXUS.h"
#include "linearSolver.h"

#include <cassert>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

#ifdef HAS_NEXUS
#include <nexus/nexus.h>
#endif /* HAS_NEXUS */

namespace linearSolver
{

template <size_t N>
class NEXUS : public ::linearSolver::Base<N>
{
public:
    using Base = ::linearSolver::Base<N>;
    using Base::BLOCKSIZE;
    using Coefficients = typename Base::Coefficients;
    using Context = typename Base::Context;
    using Index = typename Context::Index;

protected:
    using Base::ctx_;

public:
    NEXUS() = delete;

    explicit NEXUS(const YAML::Node& node, const bool precond = false)
        : Base(ID::AMG, precond)
    {
        const YAML::Node conf = node["nexus"];
        if (!conf || !conf.IsMap())
        {
            throw std::runtime_error(
                "linearSolver::NEXUS: `family: nexus` requires a `nexus:` "
                "sub-map holding the NEXUS solver configuration (family: "
                "AMGSolver, GMRES or DirectLU plus options)");
        }
        YAML::Emitter out;
        out << conf;
        config_ = out.c_str();
        device_capable_ = deviceCapable_(conf);
    }

    ~NEXUS() override
    {
#ifdef HAS_NEXUS
        NEXUS_SolverDestroy(&solver_);
#endif /* HAS_NEXUS */
    }

    std::shared_ptr<Context>
    createContext(const std::string& system_name,
                  const CRSNodeGraph* graph) override
    {
#ifdef HAS_NEXUS
        auto ctx = std::make_shared<ContextNEXUS<N>>(
            "NEXUS", system_name, graph, nullptr, device_capable_);
        ctx_ = ctx;
        ctx_nexus_ = ctx.get();
        createSolver_(system_name);
#else
        throw std::runtime_error(
            "linearSolver::NEXUS::createContext: not supported by this "
            "build");
#endif /* HAS_NEXUS */
        return ctx_;
    }

    std::shared_ptr<Context>
    createContext(const std::string& system_name,
                  std::shared_ptr<Coefficients> coeffs) override
    {
#ifdef HAS_NEXUS
        auto ctx = std::make_shared<ContextNEXUS<N>>(
            "NEXUS", system_name, coeffs, nullptr, device_capable_);
        ctx_ = ctx;
        ctx_nexus_ = ctx.get();
        createSolver_(system_name);
#else
        throw std::runtime_error(
            "linearSolver::NEXUS::createContext: not supported by this "
            "build");
#endif /* HAS_NEXUS */
        return ctx_;
    }

    int solve() override
    {
#ifdef HAS_NEXUS
        assert(ctx_);
        assert(ctx_nexus_);

        // device-resident builds stage through the attached host mirrors;
        // host builds run zero-copy on the framework storage
        ctx_nexus_->uploadSystem();
        callNEXUS_(NEXUS_SolverSolve(solver_), "NEXUS_SolverSolve");
        ctx_nexus_->downloadSolution();

        // convergence diagnostics from NEXUS (computed on its side; the
        // base-class prologue/epilogue host loops are unusable on device
        // builds and redundant here)
        auto& ctrl = ctx_->getControlData();
        int iterations = 0;
        double initial[BLOCKSIZE];
        double final_res[BLOCKSIZE];
        double scaled_initial[BLOCKSIZE];
        double scaled_final[BLOCKSIZE];
        callNEXUS_(NEXUS_SolverIterations(solver_, &iterations),
                   "NEXUS_SolverIterations");
        callNEXUS_(NEXUS_SolverResiduals(solver_, initial, final_res),
                   "NEXUS_SolverResiduals");
        callNEXUS_(
            NEXUS_SolverScaledResiduals(solver_, scaled_initial,
                                        scaled_final),
            "NEXUS_SolverScaledResiduals");
        ctrl.n_iterations = iterations;
        for (size_t k = 0; k < N; k++)
        {
            ctrl.solver_initial_res[k] = initial[k];
            ctrl.solver_final_res[k] = final_res[k];
            ctrl.scaled_initial_res[k] = scaled_initial[k];
            ctrl.scaled_final_res[k] = scaled_final[k];
        }

        ++(*ctx_); // increment solver call count
#else
        throw std::runtime_error(
            "linearSolver::NEXUS::solve: not supported by this build");
#endif /* HAS_NEXUS */
        return 0;
    }

private:
    std::string config_;
    bool device_capable_ = false;

    // A system's fine-level x/b are safe to keep device-resident only when
    // its effective AMG cycle runs on the device (v-cycle + jacobi smoother
    // + kokkos backend); the no-cycle smoother and ILU0 run on the host and
    // would read the device vectors there.  Determined once from the config.
    static bool deviceCapable_(const YAML::Node& conf)
    {
        auto lower = [](std::string s) {
            for (char& c : s)
            {
                c = static_cast<char>(std::tolower(
                    static_cast<unsigned char>(c)));
            }
            return s;
        };
        std::string family =
            conf["family"] ? lower(conf["family"].as<std::string>()) : "";
        const YAML::Node opts = conf["options"];
        if (!opts)
        {
            return false;
        }
        // the AMG node whose cycle governs where x/b are touched
        YAML::Node amg;
        if (family == "gmres")
        {
            if (!opts["backend"] ||
                lower(opts["backend"].as<std::string>()) != "kokkos")
            {
                return false; // host Krylov reads host x/b
            }
            amg = opts["precond"];
        }
        else if (family == "amgsolver")
        {
            amg = conf;
        }
        else
        {
            return false; // DirectLU etc.: host
        }
        if (!amg || !amg["options"])
        {
            return false;
        }
        const YAML::Node ao = amg["options"];
        const auto val = [&](const char* k) {
            return ao[k] ? lower(ao[k].as<std::string>()) : std::string();
        };
        // NEXUS runs the device cycle for a jacobi, ILU(0) or DILU smoother on
        // both the v-cycle and the smoother-only no-cycle
        // (kokkosCycleSupported_); each consumes the attached device values
        // with no host staging.
        const std::string cyc = val("cycle_type");
        const std::string sm = val("smoother_type");
        return val("backend") == "kokkos" &&
               (cyc == "v-cycle" || cyc == "no-cycle") &&
               (sm == "jacobi" || sm == "ilu0" || sm == "dilu");
    }

#ifdef HAS_NEXUS
    NEXUS_Solver solver_ = nullptr;
    ContextNEXUS<N>* ctx_nexus_ = nullptr;

    void createSolver_(const std::string& system_name)
    {
        callNEXUS_(NEXUS_SolverCreate(config_.c_str(), nullptr, &solver_),
                   "NEXUS_SolverCreate");
        callNEXUS_(NEXUS_SolverSetMatrix(solver_,
                                         ctx_nexus_->getMatrixNEXUS(),
                                         system_name.c_str()),
                   "NEXUS_SolverSetMatrix");
    }

    static void callNEXUS_(const int status, const char* what)
    {
        if (status != 0)
        {
            throw std::runtime_error(std::string(what) + ": " +
                                     NEXUS_ErrorMessage());
        }
    }
#endif /* HAS_NEXUS */
};

} /* namespace linearSolver */

#endif /* NEXUSSOLVER_H_LINSOLVE */
