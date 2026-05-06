// File       : AMGFactory.cpp
// Created    : Fri Apr 03 2026 10:41:51 (+0200)
// Author     : Fabian Wermelinger
// Description: AMG factory instance implementation
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.

#include "AMGFactory.h"
#include <cassert>
#include <dlfcn.h>
#include <stdexcept>

static void* handle = nullptr;

namespace linearSolver
{

AMGFactory getAMGFactory(const AMGFactoryType type)
{
    if (!handle)
    {
        handle = dlopen("libAMG.so", RTLD_NOW | RTLD_LOCAL);
        if (!handle)
        {
            throw std::runtime_error(dlerror());
        }
    }

    AMGFactory f = nullptr;
    if (AMGFactoryType::AMG == type)
    {
        f = (AMGFactory)dlsym(handle, "getAMGSolverInstance");
    }
    else if (AMGFactoryType::GMRES == type)
    {
        f = (AMGFactory)dlsym(handle, "getGMRESSolverInstance");
    }
    else if (AMGFactoryType::DIRECT == type)
    {
        f = (AMGFactory)dlsym(handle, "getDirectSolverInstance");
    }

    assert(f);
    return f;
}

} /* namespace linearSolver */
