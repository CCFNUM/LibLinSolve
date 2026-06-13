// File       : AMGFactory.cpp
// Created    : Fri Apr 03 2026 10:41:51 (+0200)
// Author     : Fabian Wermelinger
// Description: AMG factory instance implementation
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.

#include "AMGFactory.h"
#include <cassert>

#ifdef LIBLINSOLVE_STATIC
#ifdef __cplusplus
extern "C"
{
#endif
    extern linearSolver::_AMGFactory getAMGSolverInstance;
    extern linearSolver::_AMGFactory getGMRESSolverInstance;
    extern linearSolver::_AMGFactory getDirectSolverInstance;
#ifdef __cplusplus
} /* extern "C" */
#endif
#define SYMBOL(x) (x)
#else
#include <dlfcn.h>
#include <stdexcept>
static void* handle = nullptr;
#define SYMBOL(x) (linearSolver::AMGFactory) dlsym(handle, #x)
#endif /* LIBLINSOLVE_STATIC */

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
