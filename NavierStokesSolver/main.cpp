#include <iostream>
#include <armadillo>
#include <stdio.h>

#include "data.h"
#include "mesh.h"
#include "solver.h"
#include "boundary.h"
#include "integrators.h"
#include "testsuite.h"

#include "ROM.h"

//TO DO: CONVECTION JACOBIAN

constexpr double PI = 3.14159265358979323846;
constexpr bool COLLECT_DATA = false;

int main() {

	//FULL ORDER MODEL

	mesh Mesh(400, 400, 1.0, 1.0); // 2 * PI, 2 * PI);

	//solver has to check 2x (UL or LR) Periodic is given, otherwise throw error 
	solver Solver(
			Mesh,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			POISSON_SOLVER::FOURIER,
			0.00005
		);
	
	ButcherTableau tableRK4({
		4,
		{{0.0}, {1.0 / 2.0}, {0.0, 1.0 / 2.0}, {0.0, 0.0, 1.0}},
		{1.0 / 6.0,	1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 },
		{ 0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 }
		}
	);

	ExplicitRungeKutta_NS<COLLECT_DATA> RK4(tableRK4);

	double Time			= 2.0;
	double dt			= 0.001;
	double collectTime	= 8.0;

	arma::Col<double> vel	= Solver.setupTestCase(TESTSUITE::FREELY_DECAYING_2D_TURBULENCE);
	arma::Col<double> p		= arma::zeros(0.0);

	arma::Col<double> velInit = vel;

	vel = RK4.integrate(Time, dt, vel, p, Solver, collectTime);

	arma::Col<double> velInterp = Solver.interpolateVelocity(vel);

	if (remove("vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	velInterp.save("vel.txt", arma::raw_ascii);

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

	//REDUCED ORDER MODEL

	/*
	int numModesPOD = 10;

	ROM_Solver RomSolver(Solver, RK4.getDataCollector(), numModesPOD);

	ExplicitRungeKutta_ROM<false> RK4r(tableRK4);

	arma::Col<double> a		= RomSolver.calculateIC(velInit);
	arma::Col<double> aInit = a;

	a = RK4r.integrate(Time, dt, a, p, RomSolver);

	arma::Col<double> velr	= RomSolver.Psi() * a;

	arma::Col<double> RomVelInterp = Solver.interpolateVelocity(velr);

	if (remove("rom_vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	RomVelInterp.save("rom_vel.txt", arma::raw_ascii);

	std::cout << "divergence: " << (Solver.M() * velr).max() << std::endl;
	*/

	return 0;
}

