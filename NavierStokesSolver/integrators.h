#ifndef H_INTEGRATORS
#define H_INTEGRATORS

#include <iostream>
#include <armadillo>
#include <vector>
#include <cmath>
#include <type_traits>

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

template<bool COLLECT_DATA=false>
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

	//classical forward Euler
	static ButcherTableau EulerForward() { return Get().EulerForward_Impl(); }

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

	ButcherTableau EulerForward_Impl() {
		return {
					1,
				{
					{0.0}
				},
					{1.0},
					{0.0}
		};

	}

};




//class that does temporally localized model reduction
template<typename METHOD>
class spPID
{
private:
	dataCollector<true>& m_dataCollector;
	HYPER_REDUCTION_METHOD m_method;

	std::vector<int> m_numModesPod;
	std::vector<int> m_numModesDEIM;
	std::vector<int> m_numPointsDEIM;
	int m_numIntervals;
	int m_overlap;

	bool m_loadFromData_flag;
	std::vector<int> m_datasetIndices;
	std::string m_datasetName1;
	std::string m_datasetName2;

	std::vector<ROM_Solver> m_localSolvers;
	std::vector<METHOD> m_localHyperReduction;
	std::vector<arma::Mat<double>> m_transitionMatrices;


public:
	spPID(
		const solver& solver, 
		dataCollector<true>& dataCollector, 
		HYPER_REDUCTION_METHOD method, 
		const std::vector<int>& numModesPod, 
		const std::vector<int>& numModesDEIM,
		const std::vector<int>& numPointsDEIM,
		int numIntervals, 
		int overlap = 0, 
		const std::vector<int>& datasetIndices = {-1},
		const std::string& datasetName1 = "",
		const std::string& datasetName2 = ""
	)
		: m_dataCollector(dataCollector),
		m_method(method),
		m_numModesPod(numModesPod),
		m_numModesDEIM(numModesDEIM),
		m_numPointsDEIM(numPointsDEIM),
		m_numIntervals(numIntervals),
		m_transitionMatrices(numIntervals - 1),
		m_overlap(overlap),
		m_datasetIndices(datasetIndices),
		m_datasetName1(datasetName1),
		m_datasetName2(datasetName2)
	{

		if (m_datasetIndices[0] != -1)
			m_loadFromData_flag = true; 
		else 
			m_loadFromData_flag = false;

		setupHyperReduction();

		setupROMSolver(solver);

		computeInterfaceConditions(solver);

	}

	arma::Col<double> solve(const arma::Col<double>&, double, double) const;

	const std::vector<ROM_Solver>& getSolvers() const;

private:

	void setupHyperReduction();

	void setupROMSolver(const solver&);

	void computeInterfaceConditions(const solver&);

};


template<typename METHOD>
void spPID<METHOD>::setupHyperReduction() {

	if (!m_loadFromData_flag) {

		int N, n, nLast;

		N = m_dataCollector.getOperatorMatrix().n_cols;

		n = N / m_numIntervals;
		nLast = N - 1 - (m_numIntervals - 1) * (N / m_numIntervals);

		int index1, index2;

		for (int i = 0; i < m_numIntervals; ++i) {

			if (i != m_numIntervals - 1) {
				if (i != 0) {
					index1 = i * n - m_overlap;
					index2 = (i + 1) * n - 1 + m_overlap;
				}
				else {
					index1 = i * n;
					index2 = (i + 1) * n - 1 + m_overlap;
				}
			}
			else {
				index1 = i * n - m_overlap;
				index2 = i * n + nLast;
			}

			if constexpr (std::is_same<METHOD, LSDEIM>::value)
				m_localHyperReduction.emplace_back(m_numModesDEIM[i], m_numPointsDEIM[i], m_dataCollector.split(index1, index2));

			if constexpr (std::is_same<METHOD, DEIM>::value)
				m_localHyperReduction.emplace_back(m_numModesDEIM[i], m_dataCollector.split(index1, index2));

		}

	}
	else {

		std::string name;

		arma::Mat<double> dataLeft, dataRight;
		int N;

		for (int i = 0; i < m_numIntervals; ++i) {

			if (i != 0) {
				name = m_datasetName2 + std::to_string(m_datasetIndices[i - 1]);
				m_dataCollector.loadOperatorMatrix(name);

				N = m_dataCollector.getOperatorMatrix().n_cols - 1;

				dataLeft = m_dataCollector.getOperatorMatrix().cols(N - m_overlap, N);
			}

			if (i != m_numIntervals - 1) {
				name = m_datasetName2 + std::to_string(m_datasetIndices[i + 1]);
				m_dataCollector.loadOperatorMatrix(name);

				dataRight = m_dataCollector.getOperatorMatrix().cols(0, m_overlap);
			}


			name = m_datasetName2 + std::to_string(m_datasetIndices[i]);
			m_dataCollector.loadOperatorMatrix(name);

			if (i != 0)
				m_dataCollector.appendOperatorLeft(dataLeft);

			if (i != m_numIntervals - 1)
				m_dataCollector.appendOperatorRight(dataRight);

			if constexpr (std::is_same<METHOD, LSDEIM>::value)
				m_localHyperReduction.emplace_back(m_numModesDEIM[i], m_numPointsDEIM[i], m_dataCollector);

			if constexpr (std::is_same<METHOD, DEIM>::value)
				m_localHyperReduction.emplace_back(m_numModesDEIM[i], m_dataCollector);

		}

	}
}


template<typename METHOD>
void spPID<METHOD>::setupROMSolver(const solver& solver) {

	if (!m_loadFromData_flag) {

		int N, n, nLast;

		N = m_dataCollector.getDataMatrix().n_cols;

		n = N / m_numIntervals;
		nLast = N - 1 - (m_numIntervals - 1) * (N / m_numIntervals);

		for (int i = 0; i < m_numIntervals; ++i) {

			if (i != m_numIntervals - 1) {

				if (i != 0) {
					m_localSolvers.emplace_back(solver, m_dataCollector.split(i * n - m_overlap, (i + 1) * n - 1 + m_overlap), m_numModesPod[i], m_localHyperReduction[i]);
				}
				else {
					m_localSolvers.emplace_back(solver, m_dataCollector.split(i * n, (i + 1) * n - 1 + m_overlap), m_numModesPod[i], m_localHyperReduction[i]);
				}
			}
			else {
				m_localSolvers.emplace_back(solver, m_dataCollector.split(i * n - m_overlap, i * n + nLast), m_numModesPod[i], m_localHyperReduction[i]);
			}

		}

	}
	else {

		std::string name;

		arma::Mat<double> dataLeft, dataRight;
		int N;

		for (int i = 0; i < m_numIntervals; ++i) {

			if (i != 0) {
				name = m_datasetName1 + std::to_string(m_datasetIndices[i - 1]);
				m_dataCollector.loadDataMatrix(name);

				N = m_dataCollector.getDataMatrix().n_cols - 1;

				dataLeft = m_dataCollector.getDataMatrix().cols(N - m_overlap, N);
			}

			if (i != m_numIntervals - 1) {
				name = m_datasetName1 + std::to_string(m_datasetIndices[i + 1]);
				m_dataCollector.loadDataMatrix(name);

				dataRight = m_dataCollector.getDataMatrix().cols(0, m_overlap);
			}


			name = m_datasetName1 + std::to_string(m_datasetIndices[i]);
			m_dataCollector.loadDataMatrix(name);

			if (i != 0)
				m_dataCollector.appendDataLeft(dataLeft);

			if (i != m_numIntervals - 1)
				m_dataCollector.appendDataRight(dataRight);


			m_localSolvers.emplace_back(solver, m_dataCollector, m_numModesPod[i], m_localHyperReduction[i]);

		}

	}
}


template<typename METHOD>
void spPID<METHOD>::computeInterfaceConditions(const solver& solver) {

	int n_cols1, n_cols2;

	for (int i = 0; i < m_numIntervals - 1; ++i) {

		n_cols2 = m_localSolvers[i + 1].Psi().n_cols;
		n_cols1 = m_localSolvers[i].Psi().n_cols;

		m_transitionMatrices[i] = m_localSolvers[i + 1].Psi().cols(2, n_cols2 - 1).t()* solver.Om()* m_localSolvers[i].Psi().cols(2, n_cols1 - 1);
	}

}

template<typename METHOD>
arma::Col<double> spPID<METHOD>::solve(const arma::Col<double>& velInit, double Time, double dt) const {

	ExplicitRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

	arma::Col<double> a = m_localSolvers[0].calculateIC(velInit);

	arma::Col<double> p = arma::zeros(0.0);
	arma::Col<double> b, c;

	for (int i = 0; i < m_numIntervals; ++i) {

		a = RKr.integrate(Time / m_numIntervals, dt, a, p, m_localSolvers[i]);

		if (i < m_numIntervals - 1) {

			b = m_transitionMatrices[i] * a.rows(2, a.n_rows - 1);
			c = arma::norm(a.rows(2, a.n_rows - 1), 2) / arma::norm(b, 2) * b;

			c.insert_rows(0, a.rows(0, 1));

			a = c;

		}
	}

	return a;
}


template<typename METHOD>
const std::vector<ROM_Solver>& spPID<METHOD>::getSolvers() const {
	return m_localSolvers;
}


#endif
