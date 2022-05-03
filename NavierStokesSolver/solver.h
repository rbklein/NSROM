#ifndef H_SOLVER
#define H_SOLVER

#include <iostream>
#include <armadillo>
#include <utility>
#include <complex>

#include "mesh.h"
#include "boundary.h"
#include "testsuite.h"

enum class POISSON_SOLVER {
	DIRECT,
	FOURIER,
};

enum class LINEAR_SOLVER {
	DIRECT,
	BICGSTAB,
	GMRES,
	//iterative methods
};

//solves incompressible Navier-Stokes equations
class solver {
private:
	mesh& m_mesh; //verify default copy-constructor
	B_CONDITION m_bcUp, m_bcRight, m_bcLower, m_bcLeft;

	POISSON_SOLVER m_pSolver;

	int m_nCellsX, m_nCellsY;

	//diffusion matrix
	arma::SpMat<double> m_D;

	//divergence matrix
	arma::SpMat<double> m_M;

	//pressure gradient matrix
	arma::SpMat<double> m_G;

	//pressure poisson matrix
	arma::SpMat<double> m_L;

	//pressure poisson fourier transform
	arma::Mat<std::complex<double>> m_Lhat;

	//Cell Size matrices
	arma::SpMat<double> m_Omega, m_OmegaInv;

	//kinematic viscosity
	double m_nu;

public:
	solver(mesh& mesh, B_CONDITION bcUp, B_CONDITION bcRight, B_CONDITION bcLower, B_CONDITION bcLeft, POISSON_SOLVER pSolver, double nu) 
		:	m_mesh(mesh),
			m_bcUp(bcUp), m_bcRight(bcRight), m_bcLower(bcLower), m_bcLeft(bcLeft),
			m_pSolver(pSolver),
			m_nu(nu)
	{
		m_nCellsX	= m_mesh.getNumCellsX();
		m_nCellsY	= m_mesh.getNumCellsY();

		m_mesh.processBoundary(m_bcUp, m_bcRight, m_bcLower, m_bcLeft);

		//assigning like this can also be done inside the function (saves a copy constructor call. code is written for return value optimization though)
		m_D			= setupDiffusionMatrix();

		m_M			= setupDivergenceMatrix();

		m_G			= -m_M.t();

		std::pair<arma::SpMat<double>, arma::SpMat<double>> OmegaAndOmegaInv = setupOmegaMatrices();

		m_Omega		= OmegaAndOmegaInv.first;

		m_OmegaInv	= OmegaAndOmegaInv.second;

		setupPressurePoissonMatrix();

	}

	arma::Col<double>		   N(const arma::Col<double>&) const;

	//Convection Jacobian
	arma::SpMat<double>		   J(const arma::Col<double>&) const;

	const arma::SpMat<double>& D() const;
	const arma::SpMat<double>& M() const;
	const arma::SpMat<double>& G() const;
	const arma::SpMat<double>& Om() const;
	const arma::SpMat<double>& OmInv() const;
	const arma::SpMat<double>& L() const;

	double nu() const;

	arma::Col<double> setupTestCase(TESTSUITE);
	arma::Col<double> interpolateVelocity(const arma::Col<double>&) const;

	arma::Col<double> poissonSolve(const arma::Col<double>&) const;

	POISSON_SOLVER getSolverType() const;

	const mesh& getMesh() const;

	std::pair<arma::uword, arma::uword> vectorToGridIndex(arma::uword) const;

private:
	arma::SpMat<double> setupDiffusionMatrix();
	arma::SpMat<double> setupDivergenceMatrix();
	std::pair<arma::SpMat<double>, arma::SpMat<double>> setupOmegaMatrices();
	void setupPressurePoissonMatrix();
};

//consider doing this for all getters
inline POISSON_SOLVER solver::getSolverType() const {
	return m_pSolver;
}


#endif
