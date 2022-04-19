#ifndef H_ROM
#define H_ROM

#include <iostream>
#include <armadillo>

#include "solver.h"
#include "data.h"

enum class HYPER_REDUCTION_METHOD {
	NONE,
	EXACT_TENSOR_DECOMPOSITION,
	DEIM,
	SP_DEIM,
};

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

	const solver& getSolver() const;

private:

	void setupBasis();

	void precomputeOperators();

};













/*
//potential optimizations: template arguments or static polymorphism
class Base_HyperReduction {

protected:
	HYPER_REDUCTION_METHOD m_method;

	Base_HyperReduction(HYPER_REDUCTION_METHOD method)
		: m_method(method)
	{}

public:
	Base_HyperReduction(const Base_HyperReduction& other)
		: m_method(other.getMethod())
	{}

	HYPER_REDUCTION_METHOD getMethod() const;

	virtual arma::Col<double> Nr(const arma::Col<double>& a, const ROM_Solver& solver) const = 0;
};

class Naive : public Base_HyperReduction {

public:
	Naive()
		: Base_HyperReduction(HYPER_REDUCTION_METHOD::NONE)
	{}

	virtual arma::Col<double> Nr(const arma::Col<double>& a, const ROM_Solver& solver) const override;
};
*/

#endif
