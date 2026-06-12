// File       : blockMatrixIsZero.hpp
// Created    : Tue May 19 2026
// Author     : Mhamad Mahdi Alloush
// Description: block matrix exact-zero check
// Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.

// no include guard: must only be include in blockMatrixOperators.h

namespace linearSolver
{
namespace BlockMatrix
{

// Exact-zero check on a flat NxN block (row-major i*BLOCKSIZE+j).
template <int BLOCKSIZE, typename TReal>
constexpr bool isZero(const TReal* M)
{
    for (int k = 0; k < BLOCKSIZE * BLOCKSIZE; ++k)
    {
        if (M[k] != static_cast<TReal>(0.0))
        {
            return false;
        }
    }
    return true;
}

} /* namespace BlockMatrix */
} /* namespace linearSolver */
