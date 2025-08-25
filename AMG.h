// File       : AMG.h
// Created    : Fri Mar 14 2025 17:47:33 (+0100)
// Author     : Fabian Wermelinger
// Description: AMG instance factory
// Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef AMG_H_FCOHLAMO
#define AMG_H_FCOHLAMO

#ifndef NO_LINSOLVE_CONF
#include "Config.h"
#endif /* NO_LINSOLVE_CONF */
#include "CRSNodeGraph.h"
#include <yaml-cpp/yaml.h>

#if defined _WIN32 || defined __CYGWIN__
#define LIBAMG_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define LIBAMG_EXPORT __attribute__((visibility("default")))
#else
#define LIBAMG_EXPORT
#endif
#endif

namespace linearSolver
{

LIBAMG_EXPORT void* getAMGSolverInstance(const size_t blocksize,
                                         const YAML::Node& node,
                                         const YAML::Node& solver_lookup,
                                         GraphLayout& layout);
LIBAMG_EXPORT void* getGMRESSolverInstance(const size_t blocksize,
                                           const YAML::Node& node,
                                           const YAML::Node& solver_lookup,
                                           GraphLayout& layout);
LIBAMG_EXPORT void* getDirectSolverInstance(const size_t blocksize,
                                            const YAML::Node& node,
                                            GraphLayout& layout);

} /* namespace linearSolver */

#endif /* AMG_H_FCOHLAMO */
