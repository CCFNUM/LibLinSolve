// File       : AMGFactory.h
// Created    : Fri Mar 14 2025 17:47:33 (+0100)
// Author     : Fabian Wermelinger
// Description: AMG instance factory
// Copyright 2025 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef AMGFACTORY_H_WFDSJH5K
#define AMGFACTORY_H_WFDSJH5K

#ifndef NO_LINSOLVE_CONF
#include "Config.h"
#endif /* NO_LINSOLVE_CONF */
#include "CRSNodeGraph.h"
#include <yaml-cpp/yaml.h>

namespace linearSolver
{

typedef void*(_AMGFactory)(const size_t blocksize,
                           const YAML::Node& node,
                           GraphLayout& layout,
                           const YAML::Node* solver_lookup);
typedef _AMGFactory* AMGFactory;

enum class AMGFactoryType
{
    AMG,
    GMRES,
    DIRECT
};

AMGFactory getAMGFactory(const AMGFactoryType type);

} /* namespace linearSolver */

#endif /* AMGFACTORY_H_WFDSJH5K */
