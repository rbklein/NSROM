#include <iostream>
#include <armadillo>
#include <stdio.h>
#include <vector>

#include "data.h"
#include "mesh.h"
#include "solver.h"
#include "boundary.h"
#include "integrators.h"
#include "testsuite.h"

#include "ROM.h"

//TO DO: CONVECTION JACOBIAN

constexpr double PI = 3.14159265358979323846;
constexpr bool COLLECT_DATA = true;

#define FOM_CODE
#define ROM_CODE
//#define TEST_CODE_SPATIAL
//#define TEST_CODE_TEMPORAL


int main() {

	//FULL ORDER MODEL

#ifdef FOM_CODE

	mesh Mesh(100, 100, 2 * PI, 2 * PI);  //(400, 400, 1.0, 1.0);

	//solver has to check 2x (UL or LR) Periodic is given, otherwise throw error 
	solver Solver(
			Mesh,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			POISSON_SOLVER::FOURIER,
			0.001
		);
	
	ButcherTableau tableRK4({
		4,
		{{0.0}, {1.0 / 2.0}, {0.0, 1.0 / 2.0}, {0.0, 0.0, 1.0}},
		{1.0 / 6.0,	1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 },
		{ 0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 },
		}
	);

	ExplicitRungeKutta_NS<COLLECT_DATA> RK4(tableRK4);

	double Time			= 8.0;
	double dt			= 0.01;
	double collectTime	= 8.0;

	arma::Col<double> vel	= Solver.setupTestCase(TESTSUITE::SHEAR_LAYER_ROLL_UP);
	arma::Col<double> p		= arma::zeros(0.0);

	arma::Col<double> velInit = vel;

	vel = RK4.integrate(Time, dt, vel, p, Solver, collectTime);

	arma::Col<double> velInterp = Solver.interpolateVelocity(vel);

	if (remove("vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	velInterp.save("vel.txt", arma::raw_ascii);

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

#endif

	//REDUCED ORDER MODEL

#ifdef ROM_CODE

	int numModesPOD = 10;
	int numModesDEIM = 10;

	//noHyperReduction hyperReduction;

	DEIM deim(numModesDEIM, RK4.getDataCollector());

	ROM_Solver RomSolver(Solver, RK4.getDataCollector(), numModesPOD, deim);

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

#endif

	//spatial convergence test

#ifdef TEST_CODE_SPATIAL

	std::vector<double> spatialConvergence;
	std::vector<int> nCells = { 5, 10, 20, 40, 80, 160, 320 };

	for (int i = 0; i < 7; ++i) {

		mesh Mesh(nCells[i], nCells[i], 2.0 * PI, 2.0 * PI);

		solver Solver(
			Mesh,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			POISSON_SOLVER::FOURIER,
			0.01
		);

		ButcherTableau tableRK4({
			4,
			{{0.0}, {1.0 / 2.0}, {0.0, 1.0 / 2.0}, {0.0, 0.0, 1.0}},
			{1.0 / 6.0,	1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 },
			{ 0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 }
			}
		);

		ExplicitRungeKutta_NS<COLLECT_DATA> RK4(tableRK4);

		double Time = 1.0;
		double dt = 0.001;

		arma::Col<double> vel		= Solver.setupTestCase(TESTSUITE::TAYLOR_GREEN_VORTEX);
		arma::Col<double> p			= arma::zeros(0.0);
		arma::Col<double> velInit	= vel;

		vel = RK4.integrate(Time, dt, vel, p, Solver);

		arma::Col<double> exact(vel.n_elem);

		const arma::field<cell>& CellsU = Solver.getMesh().getCellsU();
		const arma::field<cell>& CellsV = Solver.getMesh().getCellsV();
		const arma::field<cell>& CellsP = Solver.getMesh().getCellsP();

		for (arma::uword q = Solver.getMesh().getStartIndUy(); q < Solver.getMesh().getEndIndUy(); ++q) {
			for (arma::uword p = Solver.getMesh().getStartIndUx(); p < Solver.getMesh().getEndIndUx(); ++p) {

				exact(CellsU(q, p).vectorIndex) = cos(CellsU(q, p).x) * sin(CellsU(q, p).y) * exp(-2.0 * Solver.nu() * Time);

			}
		}

		for (arma::uword q = Solver.getMesh().getStartIndVy(); q < Solver.getMesh().getEndIndVy(); ++q) {
			for (arma::uword p = Solver.getMesh().getStartIndVx(); p < Solver.getMesh().getEndIndVx(); ++p) {

				exact(CellsV(q, p).vectorIndex) = -1.0 * sin(CellsV(q, p).x) * cos(CellsV(q, p).y) * exp(-2.0 * Solver.nu() * Time);

			}
		}

		arma::Col<double> error = exact - vel;

		spatialConvergence.push_back(arma::norm(exact - vel, "inf"));

	}

	remove("spatial_conv.txt");

	arma::Col<double>(spatialConvergence).save("spatial_conv.txt", arma::raw_ascii);

#endif

	//temporal convergence test

#ifdef TEST_CODE_TEMPORAL

	std::vector<double> temporalConvergence;

	mesh Mesh(20, 20, 2 * PI, 2 * PI);

	solver Solver(
		Mesh,
		B_CONDITION::PERIODIC_UL,
		B_CONDITION::PERIODIC_LR,
		B_CONDITION::PERIODIC_UL,
		B_CONDITION::PERIODIC_LR,
		POISSON_SOLVER::FOURIER,
		0.1
	);

	ButcherTableau tableRK4({
			4,
			{{0.0}, {1.0 / 2.0}, {0.0, 1.0 / 2.0}, {0.0, 0.0, 1.0}},
			{1.0 / 6.0,	1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 },
			{ 0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 }
		}
	);

	std::vector<double> timeSteps = { 0.1, 0.05, 0.025, 0.0125, 0.00625 };

	double Time = 1.0;

	ExplicitRungeKutta_NS<COLLECT_DATA> RK4(tableRK4);

	arma::Col<double> vel = Solver.setupTestCase(TESTSUITE::TAYLOR_GREEN_VORTEX);
	arma::Col<double> p = arma::zeros(0.0);
	arma::Col<double> velInit = vel;

	arma::Col<double> exact(vel.n_elem);

	const arma::field<cell>& CellsU = Solver.getMesh().getCellsU();
	const arma::field<cell>& CellsV = Solver.getMesh().getCellsV();
	const arma::field<cell>& CellsP = Solver.getMesh().getCellsP();

	for (arma::uword q = Solver.getMesh().getStartIndUy(); q < Solver.getMesh().getEndIndUy(); ++q) {
		for (arma::uword p = Solver.getMesh().getStartIndUx(); p < Solver.getMesh().getEndIndUx(); ++p) {

			exact(CellsU(q, p).vectorIndex) = cos(CellsU(q, p).x) * sin(CellsU(q, p).y) * exp(-2.0 * Solver.nu() * Time);

		}
	}

	for (arma::uword q = Solver.getMesh().getStartIndVy(); q < Solver.getMesh().getEndIndVy(); ++q) {
		for (arma::uword p = Solver.getMesh().getStartIndVx(); p < Solver.getMesh().getEndIndVx(); ++p) {

			exact(CellsV(q, p).vectorIndex) = -1.0 * sin(CellsV(q, p).x) * cos(CellsV(q, p).y) * exp(-2.0 * Solver.nu() * Time);

		}
	}

	arma::Col<double> velControl = RK4.integrate(Time, 0.001, velInit, p, Solver);

	for (int i = 0; i < 5; ++i) {

		vel = RK4.integrate(Time, timeSteps[i], velInit, p, Solver);

		temporalConvergence.push_back(arma::norm(velControl - vel, "inf"));

	}

	remove("temporal_conv.txt");

	arma::Col<double>(temporalConvergence).save("temporal_conv.txt", arma::raw_ascii);

#endif


	return 0;
}

