#ifndef H_ROM
#define H_ROM

#include <iostream>
#include <armadillo>

#include "solver.h"
#include "data.h"

class ROM_Solver {

private:

	//reduced operators
	arma::Mat<double> m_Dr;

	//mode numbers
	int m_numModesPOD;

	//FOM solver reference
	const solver& m_solver;

	//truncated and structure-preserving POD basis
	arma::Mat<double> m_Psi;

	//data collector reference
	const dataCollector<true>& m_dataCollector;

public:

	ROM_Solver(const solver& solver, const dataCollector<true>& dataCollector, int numModesPOD)
		:	m_solver(solver),
			m_dataCollector(dataCollector),
			m_numModesPOD(numModesPOD)
	{
		setupBasis();

		precomputeOperators();
	}

	arma::Col<double> calculateIC(const arma::Col<double>&) const;

	const arma::Mat<double>& Dr() const;
	const arma::Mat<double>& Psi() const;

	arma::Col<double> Nr(const arma::Col<double>&) const;

	double nu() const;

private:

	void setupBasis();

	void precomputeOperators();

};

#endif
