#ifndef H_ITERATIVE
#define H_ITERATIVE

#include "armadillo"
#include "lis.h"
#include "lis_matrix.h"
#include "lis_vector.h"
#include <string>
#include <cstring>

using namespace arma;

enum class precond { none, jacobi, ilu, ssor, hybrid, is, sainv, saamg, iluc, ilut };

enum class solver_type {
    CG, BiCG, CGS, BiCGSTAB, BiCGSTAB_l, GPBiCG, TFQMR, Orthmin, GMRES, Jacobi, GaussSeidel, SOR,
    BiCG_Safe, CR, BiCR, CRS, BiCRSTAB, GPBiCR, BiCR_Safe, FGMRES, IDRs, IDRl, MINRES, COCG, COCR
};

std::string GetPreconditioner(precond preconditioner = precond::jacobi)
{
    std::string optionb4precon;
    optionb4precon = " -p ";
    switch (preconditioner)
    {
    case precond::none:
        optionb4precon += "none";
        break;
    case precond::jacobi:
        optionb4precon += "jacobi";
        break;
    case precond::ilu:
        optionb4precon += "ilu";
        break;
    case precond::ssor:
        optionb4precon += "ssor";
        break;
    case precond::hybrid:
        optionb4precon += "hybrid";
        break;
    case precond::is:
        optionb4precon += "is";
        break;
    case precond::sainv:
        optionb4precon += "sainv";
        break;
    case precond::saamg:
        optionb4precon += "saamg";
        break;
    case precond::iluc:
        optionb4precon += "iluc";
        break;
    case precond::ilut:
        optionb4precon += "ilut";
        break;
    default:
        cout << "Precoditioner not found!\n";
        throw;
        break;
    }
    return (optionb4precon);
}

std::string GetSolver(solver_type solver = solver_type::BiCG)
{
    std::string option4solver = "-i ";
    switch (solver)
    {
    case solver_type::CG:
        option4solver += "cg";
        break;
    case solver_type::BiCG:
        option4solver += "bicg";
        break;
    case solver_type::CGS:
        option4solver += "cgs";
        break;
    case solver_type::BiCGSTAB:
        option4solver += "bicgstab";
        break;
    case solver_type::BiCGSTAB_l:
        option4solver += "bicgstabl";
        break;
    case solver_type::GPBiCG:
        option4solver += "gpbicg";
        break;
    case solver_type::TFQMR:
        option4solver += "tfqmr";
        break;
    case solver_type::Orthmin:
        option4solver += "orthomin";
        break;
    case solver_type::GMRES:
        option4solver += "gmres";
        break;
    case solver_type::Jacobi:
        option4solver += "jacobi";
        break;
    case solver_type::GaussSeidel:
        option4solver += "gs";
        break;
    case solver_type::SOR:
        option4solver += "sor";
        break;
    case solver_type::BiCG_Safe:
        option4solver += "bicgsafe";
        break;
    case solver_type::CR:
        option4solver += "cr";
        break;
    case solver_type::BiCR:
        option4solver += "bicr";
        break;
    case solver_type::CRS:
        option4solver += "crs";
        break;
    case solver_type::BiCR_Safe:
        option4solver += "bicrsafe";
        break;
    case solver_type::FGMRES:
        option4solver += "fgmres";
        break;
    case solver_type::IDRs:
        option4solver += "idrs";
        break;
    case solver_type::IDRl:
        option4solver += "idrl";
        break;
    case solver_type::MINRES:
        option4solver += "minres";
        break;
    case solver_type::COCG:
        option4solver += "cocg";
        break;
    case solver_type::COCR:
        option4solver += "cocr";
        break;
    default:
        break;
    }
    return option4solver;
}

template<class vector>
void iterative_solve(vector& x, sp_mat& A, vector& b, std::string solver_tolerance = "1e-12",
    solver_type solver_typ = solver_type::GMRES, precond preconditioner = precond::ilu)
{
    A.sync();
    int argc = 0; char** argv = NULL;
    lis_initialize(&argc, &argv);


    //setup matrix
    LIS_INT n = A.n_cols;
    LIS_INT nnz = A.n_nonzero;
    LIS_MATRIX A1;

    LIS_INT* ptr = new LIS_INT[n + 1];
    LIS_INT* index = new LIS_INT[nnz];
    LIS_SCALAR* value = new LIS_SCALAR[nnz];

    for (int i = 0; i < nnz; ++i) {
        value[i] = A.values[i];
        index[i] = A.row_indices[i];
    }

    for (int i = 0; i < (n + 1); ++i) {
        ptr[i] = A.col_ptrs[i];
    }

    lis_matrix_create(0, &A1);
    lis_matrix_set_size(A1, 0, n);
    lis_matrix_set_csc(nnz, ptr, index, value, A1);
    lis_matrix_assemble(A1);


    //setup vectors
    LIS_VECTOR b1, x1;

    lis_vector_duplicate(A1, &b1);
    lis_vector_duplicate(A1, &x1);

    for (int i = 0; i < n; ++i) {
        lis_vector_set_value(LIS_INS_VALUE, i, b(i), b1);
        lis_vector_set_value(LIS_INS_VALUE, i, x(i), x1);
    }


    //setup solver
    LIS_SOLVER solver;

    std::string s_solverOption1 = GetSolver(solver_typ) + GetPreconditioner(preconditioner);;
    std::string s_solverOption2 = "-tol " + solver_tolerance + " -maxiter " + std::to_string(A.n_rows);// +" -print out";
    char* solverOption1 = const_cast<char*>(s_solverOption1.c_str());
    char* solverOption2 = const_cast<char*>(s_solverOption2.c_str());

    lis_solver_create(&solver);
    lis_solver_set_option(solverOption1, solver);
    lis_solver_set_option(solverOption2, solver);
    lis_solve(A1, b1, x1, solver);


    //copy into armadillo vector
    double val;
    for (int i = 0; i < n; ++i) {
        lis_vector_get_value(x1, i, &val);
        x(i) = val;
    }


    //free memory
    delete[] ptr;
    delete[] index;
    delete[] value;

    //lis_matrix_destroy(A1);
    //lis_vector_destroy(b1);
    //lis_vector_destroy(x1);

    lis_finalize();
}

template<class vector>
vector iterative_solve(sp_mat& A, vector& b, std::string solver_tolerance = "1e-12",
    solver_type solver_typ = solver_type::GMRES, precond preconditioner = precond::ilu)
{
    vector x;
    iterative_solve(x, A, b, solver_tolerance, solver_typ, preconditioner);
    return x;
}


#endif