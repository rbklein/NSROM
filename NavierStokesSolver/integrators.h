#ifndef H_INTEGRATORS
#define H_INTEGRATORS

#include <iostream>
#include <armadillo>
#include <vector>
#include <cmath>

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

	//b's
	std::vector<double> b;

	//c's
	std::vector<double> c;

};

template<bool COLLECT_DATA>
class ExplicitRungeKutta_NS : public Base_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;

public:

	ExplicitRungeKutta_NS(ButcherTableau tableau) 
		:	Base_Integrator<COLLECT_DATA>(), 
			m_tableau{ tableau }
	{}

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime = 0.0) override;

};

template<bool COLLECT_DATA>
class ImplicitRungeKutta_NS : public Base_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;
	LINEAR_SOLVER m_solver;

public:

	ImplicitRungeKutta_NS(ButcherTableau tableau, LINEAR_SOLVER solver)
		:	Base_Integrator<COLLECT_DATA>(), 
			m_tableau{ tableau },
			m_solver{ solver }
	{}

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime = 0.0) override;

};

template<bool COLLECT_DATA>
class RelaxationRungeKutta_NS : public Base_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;
	
public:

	RelaxationRungeKutta_NS(ButcherTableau tableau)
		:	Base_Integrator<COLLECT_DATA>(),
			m_tableau{ tableau }
	{}

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime = 0.0) override;

};



//Has to be different class from Base_Integrator because ROM_Solver and solver are not polymorphic (too much hassle with templates)
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
	const dataCollector<COLLECT_DATA>& getDataCollector() const;

};

template<bool COLLECT_DATA>
class ExplicitRungeKutta_ROM : public Base_ROM_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;

public:

	ExplicitRungeKutta_ROM(ButcherTableau tableau)
		:	Base_ROM_Integrator<COLLECT_DATA>(), 
			m_tableau{ tableau }
	{}

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime = 0.0) override;

};

template<bool COLLECT_DATA>
class ImplicitRungeKutta_ROM : public Base_ROM_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;
	LINEAR_SOLVER m_solver;

public:

	ImplicitRungeKutta_ROM(ButcherTableau tableau, LINEAR_SOLVER solver)
		: Base_ROM_Integrator<COLLECT_DATA>(),
		m_tableau{ tableau },
		m_solver{ solver }
	{}

	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime = 0.0) override;

};

template<bool COLLECT_DATA>
class RelaxationRungeKutta_ROM : public Base_ROM_Integrator<COLLECT_DATA> {

	ButcherTableau m_tableau;

public:

	RelaxationRungeKutta_ROM(ButcherTableau tableau)
		:	Base_ROM_Integrator<COLLECT_DATA>(),
			m_tableau{ tableau }
	{}
		
	virtual arma::Col<double> integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime = 0.0) override;
};

//a singleton type class providing butcher tableaus
class ButcherTableaus
{
public:

	ButcherTableaus(const ButcherTableaus&) = delete;

	static ButcherTableaus& Get()
	{
		static ButcherTableaus instance;
		return instance;
	}

	//classical Runge-Kutta 4
	static ButcherTableau RK4() { return Get().RK4_Impl(); }

	//Runge-Kutta method of order 3 with pseudo-symplectic order 6 by Aubry et al.
	static ButcherTableau RKO3PSO6() { return Get().RKO3PSO6_Impl(); }

	//Energy-conserving Gauss-Legendre implicit Runge-Kutta method of order 4
	static ButcherTableau GL4() { return Get().GL4_Impl(); }

	//Energy-conserving implicit midpoint rule of order 2
	static ButcherTableau implicitMidpoint() { return Get().implicitMidpoint_Impl(); }

private:

	ButcherTableaus() {}

	ButcherTableau RK4_Impl() {
		return {
					4,
					{	{0.0, 0.0, 0.0, 0.0},
						{1.0 / 2.0, 0.0, 0.0, 0.0}, 
						{0.0, 1.0 / 2.0, 0.0, 0.0},	
						{0.0, 0.0, 1.0, 0.0}
					},
						{1.0 / 6.0,	1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 },
						{ 0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 },
		};
	}

	ButcherTableau RKO3PSO6_Impl() {
		return {
					5,
				{	{0.0, 0.0, 0.0, 0.0, 0.0},
					{0.13502027922908531468, 0.0, 0.0, 0.0, 0.0},
					{-0.47268213605236986919, 1.05980250415418968199, 0.0, 0.0, 0.0},
					{-1.21650460595688538935, 2.16217630216752533012, -0.372345924265360030384, 0.0, 0.0},
					{0.3327444303638736757818, -0.2088266829658723128357, 1.8786561773792085608959, -1.0025739247772099238420, 0.0}
				},
					{0.04113894457091769183, 0.26732123194413937348, 0.86700906289954518480, -0.30547139552035758861, 0.13000215610575533849},
					{0.0, 0.13502027922908531468, 0.58712036810181981280, 0.57332577194527991038,1.0 }
		};
	}

	ButcherTableau GL4_Impl() {
		return {
					2,
			{		{0.25, 0.25 - 1.0 / 6.0 * sqrt(3.0)},
					{0.25 + 1.0 / 6.0 * sqrt(3.0), 0.25}
			},
					{0.5, 0.5},
					{0.5 - 1.0 / 6.0 * sqrt(3.0), 0.5 + 1.0 / 6.0 * sqrt(3.0)}
		};
	}

	ButcherTableau implicitMidpoint_Impl() {
		return {
					1,
				{	
					{0.5}
				},
					{1.0},
					{0.5}
		};
	}

};




#endif
