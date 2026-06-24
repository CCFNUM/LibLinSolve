// File       : ContextTrilinos.h
// Created    : Wed Mar 05 2025 11:04:12 (+0100)
// Author     : Mhamad Mahdi Alloush
// Description: Trilinos (Tpetra/Belos) linear solver context
// Copyright 2024 CCFNUM HSLU T&A. All Rights Reserved.
#ifndef CONTEXTTRILINOS_H_X00B2R6R
#define CONTEXTTRILINOS_H_X00B2R6R

#include "linearSolverContext.h"
#include "memoryLayout.h"

#ifdef HAS_TRILINOS
#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_Factory.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>
#endif /* HAS_TRILINOS */

namespace linearSolver
{

#ifdef HAS_TRILINOS
namespace details
{
inline std::string toLower(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline bool isBoolString(const std::string& value)
{
    const std::string key = toLower(value);
    return key == "true" || key == "false" || key == "yes" || key == "no" ||
           key == "on" || key == "off";
}

inline std::string mapBelosSolver(const std::string& value)
{
    const std::string key = toLower(value);
    if (key == "cg")
        return "CG";
    if (key == "bicgstab")
        return "BiCGStab";
    if (key == "tfqmr")
        return "TFQMR";
    if (key == "lsqr")
        return "LSQR";
    return "GMRES";
}

inline std::string mapIfpackPreconditioner(const std::string& value)
{
    const std::string key = toLower(value);
    if (key == "ilu" || key == "rilu")
        return "ILUT";
    if (key == "riluk")
        return "RILUK";
    if (key == "jacobi" || key == "relaxation")
        return "RELAXATION";
    if (key == "chebyshev")
        return "CHEBYSHEV";
    return "RELAXATION";
}
} // namespace details

template <size_t N>
class ContextTrilinos : public Context<N>
{
public:
    using Coefficients = typename Context<N>::Coefficients;
    using Matrix = typename Coefficients::Matrix;
    using Vector = typename Coefficients::Vector;
    using Index = typename Coefficients::Index;
    using DataType = typename Matrix::DataType;

private:
    // Trilinos specific types
    struct Trilinos
    {
        using LOrdinal = int;  // Tpetra local ordinal type
        using GOrdinal = long; // Tpetra local ordinal type

        // FIXME [faw 2026-06-12]: check the difference between these two node
        // types:
        // using Node = Kokkos::DefaultExecutionSpace::device_type;
        using Node = Tpetra::Map<>::node_type;
        using Map = Tpetra::Map<LOrdinal, GOrdinal, Node>;

        using OP = Tpetra::Operator<DataType, LOrdinal, GOrdinal, Node>;
        using PC = Ifpack2::Preconditioner<DataType, LOrdinal, GOrdinal, Node>;
        using Matrix = Tpetra::CrsMatrix<DataType, LOrdinal, GOrdinal, Node>;
        using Vector = Tpetra::Vector<DataType, LOrdinal, GOrdinal, Node>;
        using MVector = Tpetra::MultiVector<DataType, LOrdinal, GOrdinal, Node>;
        using Problem = Belos::LinearProblem<DataType, MVector, OP>;
        using Solver = Belos::SolverManager<DataType, MVector, OP>;
        using SolverFactory = Belos::SolverFactory<DataType, MVector, OP>;
        using PCFactory = Ifpack2::Factory;
    };

    struct LocalSystem
    {
        using crs_size_t = typename CRSRowMatrixType::size_type;
        using crs_rowmap_t =
            typename CRSRowMatrixType::row_map_type::non_const_type;
        using crs_entries_t =
            typename CRSRowMatrixType::index_type::non_const_type;
        using crs_values_t =
            typename CRSRowMatrixType::values_type::non_const_type;
        using crs_exe_space_t = typename CRSRowMatrixType::execution_space;
        using policy_t =
            typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using member_t = typename policy_t::member_type;
    };

public:
    ContextTrilinos(const std::string& family,
                    const std::string& system_name,
                    const CRSNodeGraph* graph,
                    const YAML::Node* node = nullptr)
        : Context<N>(linearSolver::ID::Trilinos,
                     family,
                     system_name,
                     graph,
                     node),
          use_preconditioner_(false), belos_solver_name_("GMRES"),
          preconditioner_type_("RELAXATION")
    {
        setup_(node);
    }

    ContextTrilinos(const std::string& family,
                    const std::string& system_name,
                    std::shared_ptr<Coefficients> coeffs,
                    const YAML::Node* node = nullptr)
        : Context<N>(linearSolver::ID::Trilinos,
                     family,
                     system_name,
                     coeffs,
                     node),
          use_preconditioner_(false), belos_solver_name_("GMRES"),
          preconditioner_type_("RELAXATION")
    {
        setup_(node);
    }

    ~ContextTrilinos()
    {
        destroySystem_();
    }

    void solvePrologue(const int solver_id, const bool preconditioner) override
    {
        // NOTE [faw 2026-06-24]: this assumes the connectivity of the matrix
        // does not change (otherwise all of the system has to be rebuilt every
        // non-linear iteration
        copyLocalToTpetra_();

        if (use_preconditioner_)
        {
            preconditioner_->compute();
        }

        Context<N>::solvePrologue(solver_id, preconditioner);
    }

    void solveEpilogue(const int solver_id, const bool preconditioner) override
    {
        copyTpetraToLocal_();
        Context<N>::solveEpilogue(solver_id, preconditioner);
    }

    std::string info(std::ostream& os,
                     const char* prefix = nullptr) const override
    {
        const std::string pfx = Context<N>::info(os, prefix);
        os << pfx << '\t' << "Belos solver:     " << belos_solver_name_ << '\n';
        os << pfx << '\t' << "Preconditioner:   "
           << (use_preconditioner_ ? preconditioner_type_ : std::string("none"))
           << '\n';
        return pfx;
    }

    int solve()
    {
        if (belos_problem_.is_null())
        {
            belos_problem_ = Teuchos::rcp(
                new typename Trilinos::Problem(matrix_, x_vec_, b_vec_));
        }
        belos_problem_->setProblem();
        if (use_preconditioner_ && !preconditioner_.is_null())
        {
            belos_problem_->setRightPrec(preconditioner_);
        }
        else
        {
            belos_problem_->setRightPrec(Teuchos::null);
        }

        if (belos_solver_.is_null())
        {
            createBelosSolver_();
        }
        belos_solver_->setProblem(belos_problem_);
        const Belos::ReturnType result = belos_solver_->solve();

        if (result != Belos::Converged && this->verbose() > 0 &&
            this->commRank() == 0)
        {
            this->cout() << "Belos solver '" << belos_solver_name_
                         << "' did not converge after "
                         << belos_solver_->getNumIters() << " iterations.\n";
        }
        return static_cast<int>(belos_solver_->getNumIters());
    }

protected:
    inline ContextTrilinos* castContextTrilinos_() override
    {
        return this;
    }

private:
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;

    Teuchos::RCP<const typename Trilinos::Map> map_;
    Teuchos::RCP<typename Trilinos::Matrix> matrix_;
    Teuchos::RCP<typename Trilinos::Vector> x_vec_;
    Teuchos::RCP<typename Trilinos::Vector> b_vec_;
    Teuchos::RCP<typename Trilinos::PC> preconditioner_;
    Teuchos::RCP<typename Trilinos::Problem> belos_problem_;
    Teuchos::RCP<typename Trilinos::Solver> belos_solver_;

    Teuchos::RCP<Teuchos::ParameterList> belos_params_;
    Teuchos::ParameterList preconditioner_params_;

    // local CRS row-based storage
    typename LocalSystem::crs_rowmap_t rowcrs_row_ptr_;
    typename LocalSystem::crs_entries_t rowcrs_indices_;
    typename LocalSystem::crs_values_t rowcrs_values_;

    bool use_preconditioner_;
    std::string belos_solver_name_;
    std::string preconditioner_type_;

    void setup_(const YAML::Node* node)
    {
        configureFromYaml_(node);
        buildCommunicator_();
        buildMap_();
        createSystem_();
    }

    void configureFromYaml_(const YAML::Node* node)
    {
        belos_params_ = Teuchos::parameterList("Belos");
        belos_params_->set("Maximum Iterations", this->max_iterations_);
        belos_params_->set("Convergence Tolerance", this->rtol_);

        if (node)
        {
            Context<N>::setOptions_(node);
            const YAML::Node& s = *node;
            if (s["options"])
            {
                const YAML::Node& opts = s["options"];
                if (opts["belos_solver"])
                {
                    belos_solver_name_ = details::mapBelosSolver(
                        opts["belos_solver"].template as<std::string>());
                }
                if (opts["belos_parameters"])
                {
                    const auto& belos_opt = opts["belos_parameters"];
                    for (const auto& entry : belos_opt)
                    {
                        const std::string key =
                            entry.first.template as<std::string>();
                        const std::string value =
                            entry.second.template as<std::string>();
                        belos_params_->set(key, value);
                    }
                }
                if (opts["preconditioner"])
                {
                    const std::string prec = details::toLower(
                        opts["preconditioner"].template as<std::string>());
                    if (prec == "none")
                    {
                        use_preconditioner_ = false;
                    }
                    else
                    {
                        use_preconditioner_ = true;
                        preconditioner_type_ =
                            details::mapIfpackPreconditioner(prec);
                    }
                }
                if (opts["preconditioner_parameters"])
                {
                    const auto& prec = opts["preconditioner_parameters"];
                    for (const auto& entry : prec)
                    {
                        const std::string key =
                            entry.first.template as<std::string>();
                        addPreconditionerParameter_(key, entry.second);
                    }
                }
            }
        }
    }

    void buildCommunicator_()
    {
        if (comm_.is_null())
        {
            comm_ = Teuchos::rcp(new Teuchos::MpiComm<int>(
                Teuchos::opaqueWrapper(this->getCommunicator())));
        }
    }

    void buildMap_()
    {
        const Matrix& A = this->getAMatrix();
        const Index n_local = A.nRows();

        Teuchos::Array<typename Trilinos::GOrdinal> gids(n_local * N);
        for (Index i = 0; i < n_local; ++i)
        {
            const Index gid = A.localToGlobal(i);
            for (Index k = 0; k < static_cast<Index>(N); ++k)
            {
                gids[i * N + k] =
                    static_cast<typename Trilinos::GOrdinal>(gid * N + k);
            }
        }
        map_ = Teuchos::rcp(new typename Trilinos::Map(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            gids(),
            0,
            comm_));
    }

    void createSystem_()
    {
        using crs_size_t = typename LocalSystem::crs_size_t;
        using crs_exe_space_t = typename LocalSystem::crs_exe_space_t;

        static_assert(
            std::is_same_v<crs_exe_space_t,
                           typename CRSMatrixType::execution_space>,
            "BsrMatrix and CsrMatrix must be in same execution space");

        static_assert(std::is_same_v<CRSMatrixType, CRSBlockMatrixType>,
                      "getAMatrix().getCrsMatrix() must be a BsrMatrix");

        const CRSMatrixType block_A = this->getAMatrix().getCrsMatrix();
        const crs_size_t blockDim = block_A.blockDim();
        const crs_size_t numBlockRows = block_A.numRows();
        const crs_size_t numBlockCols = block_A.numCols();
        const crs_size_t numBlockEntries = block_A.nnz();
        const crs_size_t numCrsRows = numBlockRows * blockDim;
        const crs_size_t numCrsCols = numBlockCols * blockDim;
        const crs_size_t numCrsEntries = numBlockEntries * blockDim * blockDim;

        rowcrs_row_ptr_ =
            typename LocalSystem::crs_rowmap_t{"crsRowMap", numCrsRows + 1};
        rowcrs_indices_ =
            typename LocalSystem::crs_entries_t{"crsEntries", numCrsEntries};
        rowcrs_values_ =
            typename LocalSystem::crs_values_t{"crsValues", numCrsEntries};

        matrix_ = Teuchos::rcp(
            new typename Trilinos::Matrix(map_, map_, this->convertToCrs_()));

        // create tpetra vectors
        x_vec_ = Teuchos::rcp(new typename Trilinos::Vector(map_));
        b_vec_ = Teuchos::rcp(new typename Trilinos::Vector(map_));

        if (use_preconditioner_)
        {
            if (preconditioner_type_ == "RELAXATION" &&
                !preconditioner_params_.isParameter("relaxation: type"))
            {
                preconditioner_params_.set("relaxation: type", "Jacobi");
            }
            createPreconditioner_();
        }
    }

    void destroySystem_()
    {
        belos_solver_ = Teuchos::null;
        belos_problem_ = Teuchos::null;
        preconditioner_ = Teuchos::null;
        rowcrs_row_ptr_ = typename LocalSystem::crs_rowmap_t{};
        rowcrs_indices_ = typename LocalSystem::crs_entries_t{};
        rowcrs_values_ = typename LocalSystem::crs_values_t{};
        matrix_ = Teuchos::null;
        x_vec_ = Teuchos::null;
        b_vec_ = Teuchos::null;
        map_ = Teuchos::null;
        comm_ = Teuchos::null;
    }

    void createBelosSolver_()
    {
        typename Trilinos::SolverFactory factory;
        belos_solver_ = factory.create(belos_solver_name_, belos_params_);
    }

    void createPreconditioner_()
    {
        typename Trilinos::PCFactory factory;
        preconditioner_ = factory.template create<typename Trilinos::Matrix>(
            preconditioner_type_.empty() ? "RELAXATION" : preconditioner_type_,
            matrix_);
        preconditioner_->setParameters(preconditioner_params_);
        preconditioner_->initialize();
    }

    void addPreconditionerParameter_(const std::string& key,
                                     const YAML::Node& value)
    {
        if (!value.IsScalar())
        {
            return;
        }

        const std::string scalar = value.Scalar();
        if (details::isBoolString(scalar))
        {
            preconditioner_params_.set(key, value.template as<bool>());
            return;
        }

        try
        {
            preconditioner_params_.set(key, value.template as<int>());
            return;
        }
        catch (const YAML::BadConversion&)
        {
        }

        try
        {
            preconditioner_params_.set(key, value.template as<double>());
            return;
        }
        catch (const YAML::BadConversion&)
        {
        }

        preconditioner_params_.set(key, scalar);
    }

    void copyLocalToTpetra_()
    {
        // A
        {
            matrix_->resumeFill();
            auto dst = matrix_->getLocalMatrixDevice().values;
            auto src = this->convertToCrs_().values;
            Kokkos::deep_copy(dst, src);
            matrix_->fillComplete();
        }

        // NOTE [faw 2026-06-12]: cannot use zero-copy paradigm for Tpetra
        // vectors because internally they are stored in a multi-vector layout
        // which is not a simple 1D layout.
        //
        // RHS b
        const Vector& src = this->coeffs_->getBVector();
        auto b = b_vec_->getLocalViewDevice(Tpetra::Access::OverwriteAll);
        Kokkos::deep_copy(Kokkos::subview(b, Kokkos::ALL, 0), src);

        // solution x
        x_vec_->putScalar(0.0);
    }

    void copyTpetraToLocal_()
    {
        const auto src = x_vec_->getLocalViewDevice(Tpetra::Access::ReadOnly);
        Vector& x = this->coeffs_->getXVector();
        Kokkos::deep_copy(x, Kokkos::subview(src, Kokkos::ALL, 0));
    }

    // NOTE [faw 2026-06-22]: Code taken from Kokkos 5.0.0 (Kokkos 4.7.X
    // does not have this method yet)
    CRSRowMatrixType convertToCrs_() const
    {
        using crs_size_t = typename LocalSystem::crs_size_t;

        const CRSMatrixType block_A = this->getAMatrix().getCrsMatrix();

        // Get size/dimension info from this Bsr. We will use crs_size_t for all
        // int types
        const crs_size_t blockDim = block_A.blockDim();
        const crs_size_t blockSize = blockDim * blockDim;
        const crs_size_t numBlockRows = block_A.numRows();
        const crs_size_t numBlockCols = block_A.numCols();
        const crs_size_t numBlockEntries = block_A.nnz();

        // Get graph and values from this Bsr
        const auto blockRowMap = block_A.graph.row_map;
        const auto blockEntries = block_A.graph.entries;
        const auto blockValues = block_A.values;

        // Compute Csr row/col/entry sizes by multiplying Bsr sizes by block
        // dimension
        const crs_size_t numCrsRows = numBlockRows * blockDim;
        const crs_size_t numCrsCols = numBlockCols * blockDim;
        const crs_size_t numCrsEntries = numBlockEntries * blockSize;

        // Create the policy, we have 3 levels of parallelism available in the
        // algorithm
        const crs_size_t maxvec = LocalSystem::policy_t::vector_length_max();
        const crs_size_t veclen = (blockDim <= maxvec) ? blockDim : maxvec;
        typename LocalSystem::policy_t policy(
            numBlockRows, Kokkos::AUTO(), veclen);

        // Fill CrsMatrix row map, entries, and values
        Kokkos::parallel_for(
            "ConvertBsrToCrs",
            policy,
            KOKKOS_LAMBDA(const typename LocalSystem::member_t& team) {
                const crs_size_t blockRow = team.league_rank();
                const crs_size_t blockRowStart = blockRowMap(blockRow);
                const crs_size_t blockRowEnd = blockRowMap(blockRow + 1);
                const crs_size_t blockRowCount = blockRowEnd - blockRowStart;

                // Iterate over block entries in this row.
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, blockRowStart, blockRowEnd),
                    [&](const crs_size_t& blockNnz)
                {
                    const crs_size_t blockCol = blockEntries(blockNnz);
                    const crs_size_t blockNum = blockNnz - blockRowStart;

                    // Iterate over block dim to get the unblocked rows
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorRange(team, blockDim),
                        [&](const crs_size_t& blockRowOffset)
                    {
                        const crs_size_t crsRow =
                            blockRow * blockDim + blockRowOffset;
                        // Each unblocked row has blockRowCount * blockDim items
                        const crs_size_t crsRowStart =
                            blockRowStart * blockSize +
                            blockRowCount * blockDim * blockRowOffset;
                        rowcrs_row_ptr_(crsRow) = crsRowStart;

                        // Iterate over block dim to get the unblocked cols
                        for (crs_size_t blockColOffset = 0;
                             blockColOffset < blockDim;
                             ++blockColOffset)
                        {
                            const crs_size_t crsCol =
                                blockCol * blockDim + blockColOffset;
                            const crs_size_t crsNnz = crsRowStart +
                                                      blockNum * blockDim +
                                                      blockColOffset;
                            rowcrs_indices_(crsNnz) = crsCol;
                            rowcrs_values_(crsNnz) = blockValues(
                                blockNnz * blockSize +
                                blockRowOffset * blockDim + blockColOffset);
                        }
                    });
                });

                // Finalize CrsMatrix row map
                if (blockRow == numBlockRows - 1)
                {
                    rowcrs_row_ptr_(numCrsRows) =
                        blockRowMap(numBlockRows) * blockSize;
                }
            });

        // Construct CrsMatrix
        return CRSRowMatrixType("convertedFromBsrMatrix",
                                numCrsRows,
                                numCrsCols,
                                rowcrs_indices_.extent(0),
                                rowcrs_values_,
                                rowcrs_row_ptr_,
                                rowcrs_indices_);
    }
};

#else

template <size_t N>
class ContextTrilinos
{
};

#endif /* HAS_TRILINOS */

} /* namespace linearSolver */

#endif /* CONTEXTTRILINOS_H_X00B2R6R */
