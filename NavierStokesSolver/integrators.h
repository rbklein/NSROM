#ifndef H_INTEGRATORS
#define H_INTEGRATORS

#include <iostream>
#include <armadillo>
#include <vector>

#include "data.h"
#include "solver.h"
#include "ROM.h"

template<bool COLLECT_DATA>
class Base_Integrator {

protected:
	dataCollector<COLLECT_DATA> m_collector;

	Base_Integrator()
		: m_collector()
	{ }

public:

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime = 0.0) = 0;

	//make this so that you can only call it with if constexpr (COLLECT_DATA == true) else throw error
	const dataCollector<COLLECT_DATA>& getDataCollector() const;

};

struct ButcherTableau {

	//RK stages
	int s;

	//a's
	std::vector<std::vector<double>> A;

	//c's
	std::vector<double> c;

	//b's
	std::vector<double> b;

};

template<bool COLLECT_DATA>
class ExplicitRungeKutta_NS : public Base_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;

public:

	ExplicitRungeKutta_NS(ButcherTableau tableau) 
		: Base_Integrator<COLLECT_DATA>(), m_tableau{ tableau }
	{}

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime = 0.0) override;

};








template<bool COLLECT_DATA>
class Base_ROM_Integrator {

protected:
	dataCollector<COLLECT_DATA> m_collector;

	Base_ROM_Integrator()
		: m_collector()
	{ }

public:

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime = 0.0) = 0;

	//make this so that you can only call it with if constexpr (COLLECT_DATA == true) else throw error
	//const dataCollector<COLLECT_DATA>& getDataCollector() const;

};

template<bool COLLECT_DATA>
class ExplicitRungeKutta_ROM : public Base_ROM_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;

public:

	ExplicitRungeKutta_ROM(ButcherTableau tableau)
		: Base_ROM_Integrator<COLLECT_DATA>(), m_tableau{ tableau }
	{}

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime = 0.0) override;

};





#endif
