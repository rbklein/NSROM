#include <iostream>
#include <armadillo>
#include <vector>

#include "data.h"
#include "solver.h"
#include "integrators.h"

template<bool COLLECT_DATA>
arma::Col<double> ExplicitRungeKutta_NS<COLLECT_DATA>::integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime) {

	std::vector<arma::Col<double>> Us;
	std::vector<arma::Col<double>> Fs;

	arma::Col<double> Vo = initialVel;
	arma::Col<double> V;
	arma::Col<double> MV;
	arma::Col<double> phi;

	double nu = solver.nu();
	double t = 0.0;

	while (t < finalT) {

		Us.push_back(Vo);

		for (int i = 0; i < (m_tableau.s - 1); ++i) {

			V = Vo;

			Fs.push_back(solver.OmInv() * (-solver.N(Us[i]) + nu * solver.D() * Us[i]));

			for (int j = 0; j < (i + 1); ++j) {

				if (i != (m_tableau.s - 2))
					V += dt * m_tableau.A[i + 1][j] * Fs[j];
				else
					V += dt * m_tableau.b[j] * Fs[j];

			}

			MV = solver.M() * V;
			
			//this is faster than having to allocate a new arma::Col<double> in solver::poissonSolve and allows us to pass MV as const reference
			if (solver.getSolverType() != POISSON_SOLVER::FOURIER)
				MV(0) = 0.0;

			phi = solver.poissonSolve(MV);

			Us.push_back(V - solver.OmInv() * solver.G() * phi);

		}

		Vo = std::move(Us.back());

		if constexpr (Base_Integrator<COLLECT_DATA>::m_collector.COLLECT_DATA) {
			if (t < collectTime)
				Base_Integrator<COLLECT_DATA>::m_collector.addColumn(Vo);
		}
			

		Us.clear();
		Fs.clear();

		t = t + dt;

		//std::cout << t << std::endl;

		//std::cout << Vo.max() << std::endl;

	}

	return Vo;
}

template arma::Col<double> ExplicitRungeKutta_NS<false>::integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime);
template arma::Col<double> ExplicitRungeKutta_NS<true>::integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const solver& solver, double collectTime);

template<bool COLLECT_DATA>
const dataCollector<COLLECT_DATA>& Base_Integrator<COLLECT_DATA>::getDataCollector() const {
	return m_collector;
}

template const dataCollector<true>& Base_Integrator<true>::getDataCollector() const;
template const dataCollector<false>& Base_Integrator<false>::getDataCollector() const;


template<bool COLLECT_DATA>
arma::Col<double> ExplicitRungeKutta_ROM<COLLECT_DATA>::integrate(double finalT, double dt, const arma::Col<double>& initialA, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime) {

	std::vector<arma::Col<double>> as;
	std::vector<arma::Col<double>> Fs;

	arma::Col<double> ao = initialA;
	arma::Col<double> a;

	double nu = solver.nu();
	double t = 0.0;

	while (t < finalT) {

		as.push_back(ao);

		for (int i = 0; i < (m_tableau.s - 1); ++i) {

			a = ao;

			Fs.push_back((- solver.Nr(as[i]) + nu * solver.Dr() * as[i]));

			for (int j = 0; j < (i + 1); ++j) {

				if (i != (m_tableau.s - 2))
					a += dt * m_tableau.A[i + 1][j] * Fs[j];
				else
					a += dt * m_tableau.b[j] * Fs[j];

			}

			as.push_back(a);
		}

		ao = as.back();

		if constexpr (Base_ROM_Integrator<COLLECT_DATA>::m_collector.COLLECT_DATA) {
			if (t < collectTime)
				Base_ROM_Integrator<COLLECT_DATA>::m_collector.addColumn(ao);
		}


		as.clear();
		Fs.clear();

		t = t + dt;

		//std::cout << t << std::endl;
	}

	return ao;
}

template arma::Col<double> ExplicitRungeKutta_ROM<false>::integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime);
template arma::Col<double> ExplicitRungeKutta_ROM<true>::integrate(double finalT, double dt, const arma::Col<double>& initialVel, const arma::Col<double>& initialP, const ROM_Solver& solver, double collectTime);

