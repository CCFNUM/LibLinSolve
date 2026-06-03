// File       : AMGFactory.cpp
// Created    : Fri Apr 03 2026 10:41:51 (+0200)
// Author     : Fabian Wermelinger
// Description: AMG factory instance implementation
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.

#include "AMGFactory.h"
#include <cassert>

#ifdef LIBLINSOLVE_STATIC
#ifdef USE_AMG_STUBS
#include <stdexcept>

void* getAMGSolverInstance(const size_t,
                           const YAML::Node&,
                           linearSolver::GraphLayout&,
                           const YAML::Node*)
{
    throw std::runtime_error(
        "This liblinsolve build does not support `getAMGSolverInstance`");
    return nullptr;
}

void* getGMRESSolverInstance(const size_t,
                             const YAML::Node&,
                             linearSolver::GraphLayout&,
                             const YAML::Node*)
{
    throw std::runtime_error(
        "This liblinsolve build does not support `getGMRESSolverInstance`");
    return nullptr;
}

void* getDirectSolverInstance(const size_t,
                              const YAML::Node&,
                              linearSolver::GraphLayout&,
                              const YAML::Node*)
{
    throw std::runtime_error(
        "This liblinsolve build does not support `getDirectSolverInstance`");
    return nullptr;
}
#else
extern linearSolver::AMGFactory getAMGSolverInstance;
extern linearSolver::AMGFactory getGMRESSolverInstance;
extern linearSolver::AMGFactory getDirectSolverInstance;
#endif /* USE_AMG_STUBS */
#define SYMBOL(x) (x)
#else
#include <dlfcn.h>
#include <stdexcept>
static void* handle = nullptr;
#define SYMBOL(x) (linearSolver::AMGFactory) dlsym(handle, #x)
#endif /* LIBLINSOLVE_INC */

namespace linearSolver
{

AMGFactory getAMGFactory(const AMGFactoryType type)
{
#ifndef LIBLINSOLVE_STATIC
    if (!handle)
    {
        handle = dlopen("libAMG.so", RTLD_NOW | RTLD_LOCAL);
        if (!handle)
        {
            throw std::runtime_error(dlerror());
        }
    }
#endif /* LIBLINSOLVE_STATIC */

    AMGFactory f = nullptr;
    if (AMGFactoryType::AMG == type)
    {
        f = SYMBOL(getAMGSolverInstance);
    }
    else if (AMGFactoryType::GMRES == type)
    {
        f = SYMBOL(getGMRESSolverInstance);
    }
    else if (AMGFactoryType::DIRECT == type)
    {
        f = SYMBOL(getDirectSolverInstance);
    }

    assert(f);
    return f;
}

} /* namespace linearSolver */

#undef SYMBOL
