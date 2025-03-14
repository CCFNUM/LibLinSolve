// File       : AMG.h
// Created    : Fri Mar 14 2025 17:47:33 (+0100)
// Author     : Fabian Wermelinger
// Description: AMG instance factory
// Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef AMG_H_FCOHLAMO
#define AMG_H_FCOHLAMO

#include "CRSNodeGraph.h"
#include <yaml-cpp/yaml.h>

namespace linearSolver
{

void* getAMGSolverInstance(const size_t blocksize,
                           const YAML::Node& node,
                           const YAML::Node& solver_lookup,
                           GraphLayout& layout);
void* getGMRESSolverInstance(const size_t blocksize,
                             const YAML::Node& node,
                             const YAML::Node& solver_lookup,
                             GraphLayout& layout);

} /* namespace linearSolver */

#endif /* AMG_H_FCOHLAMO */
