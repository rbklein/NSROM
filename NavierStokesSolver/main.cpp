#include <iostream>
#include <armadillo>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>

#include "data.h"
#include "mesh.h"
#include "solver.h"
#include "boundary.h"
#include "integrators.h"
#include "testsuite.h"
#include "plot.h"

#include "ROM.h"

//TO DO: mesh does not work for 2x2 (but who cares?)

constexpr double PI = 3.14159265358979323846;
constexpr bool COLLECT_DATA = false;
constexpr bool SAVE_DATA = false;

#define FOM_CODE
#define PIDROM_CODE
//#define ROM_CODE
//#define TEST_CODE_SPATIAL
//#define TEST_CODE_TEMPORAL
//#define TEST_CODE_DEIM_CONVERGENCE
//#define TEST_CODE_DEIM_CONVERGENCE2
//#define ERROR_ANALYSIS1

int main() {

	//FULL ORDER MODEL	

#ifdef FOM_CODE

	mesh Mesh(256, 256, 2.0 * PI, 2.0 * PI);
	
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

	ExplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	//ImplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::implicitMidpoint(), LINEAR_SOLVER::DIRECT);

	//RelaxationRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	double Time = 8.0;
	double dt = 0.01; 
	double collectTime	= 8.0;

	arma::Col<double> vel, velInit, vorticity, streamFunc;
	arma::Col<double> p			= arma::zeros(0.0);
	velInit						= Solver.setupTestCase(TESTSUITE::SHEAR_LAYER_ROLL_UP);

	vel = velInit;

	vel = RK.integrate(Time, dt, velInit, p, Solver, collectTime);	

	//vel.load("final_vel_comp1000", arma::arma_binary);
	//vel.load("final_vel_comp100", arma::arma_binary);

	plot(Solver, vel);

#endif

	//TEMPORAL LOCALIZATION REDUCED ORDER MODEL

#ifdef PIDROM_CODE

	dataCollector<true> collector;

	spPID<LSDEIM> pid(	
		Solver, 
		collector,                            //const ref?
		HYPER_REDUCTION_METHOD::LSDEIM, 
		{ 30, 33, 36 }, 
		{ 40, 40, 40 }, 
		{ 120, 120, 120 }, 
		3,
		50,
		{1,2,3},
		"poddata",
		"deimdata"
	);

	arma::Col<double> a = pid.solve(velInit, Time, dt);

	arma::Col<double> velr = pid.getSolvers()[2].Psi() * a;

	plot(Solver, velr, true);
	
#endif

	//REDUCED ORDER MODEL

#ifdef ROM_CODE

	int numModesPOD = 30;
	int numModesDEIM = 40;  //5, 10, 20, 40, 80
	int numPointsDEIM = 200; // 900; //5, 10, 20, 40, 80

	//noHyperReduction hyperReduction;

	//DEIM hyperReduction(numModesDEIM, RK.getDataCollector(), SAVE_DATA);

	//SPDEIM hyperReduction(numModesDEIM, RK.getDataCollector(), SAVE_DATA);

	LSDEIM hyperReduction(numModesDEIM, numPointsDEIM, RK.getDataCollector(),SAVE_DATA);

	ROM_Solver RomSolver(Solver, RK.getDataCollector(), numModesPOD, hyperReduction, SAVE_DATA);

	ExplicitRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

	//ImplicitRungeKutta_ROM<false> RKr(ButcherTableaus::implicitMidpoint(), LINEAR_SOLVER::DIRECT);

	//RelaxationRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

	arma::Col<double> a		= RomSolver.calculateIC(velInit);
	arma::Col<double> aInit = a;
	
	a = RKr.integrate(Time, dt, aInit, p, RomSolver);

	arma::Col<double> velr	= RomSolver.Psi() * a;

	plot(Solver, velr, true);

	std::cout << a.t() * hyperReduction.Nrh(a, RomSolver) << std::endl;

#endif

	//spatial convergence test

#ifdef TEST_CODE_SPATIAL

	std::vector<double> spatialConvergence;
	//std::vector<int> nCells = { 5, 10, 20, 40, 80, 160, 320 };
	std::vector<int> nCells = { 4, 8, 16, 32, 64, 128, 256 };


	for (int i = 0; i < 7; ++i) {

		mesh Mesh(nCells[i] - 1, nCells[i] - 1, 2.0 * PI, 2.0 * PI);

		solver Solver(
			Mesh,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			POISSON_SOLVER::FOURIER,
			0.01
		);

		ExplicitRungeKutta_NS<COLLECT_DATA> RK4(ButcherTableaus::RK4());

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

	std::vector<double> timeSteps = { 0.1, 0.05, 0.025, 0.0125, 0.00625 };

	double Time = 0.5;

	ImplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::GL4(), LINEAR_SOLVER::DIRECT);

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

	arma::Col<double> velControl = RK.integrate(Time, 0.001, velInit, p, Solver);

	for (int i = 0; i < 5; ++i) {

		vel = RK.integrate(Time, timeSteps[i], velInit, p, Solver);

		temporalConvergence.push_back(arma::norm(velControl - vel, "inf"));

	}

	remove("temporal_conv.txt");

	arma::Col<double>(temporalConvergence).save("temporal_conv.txt", arma::raw_ascii);

#endif

	//testing the error behaviour of the DEIM algorithm

#ifdef TEST_CODE_DEIM_CONVERGENCE

	mesh Mesh(256, 256, 2 * PI, 2 * PI); //(400, 400, 1.0, 1.0); 

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

	ExplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	//ImplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::implicitMidpoint(), LINEAR_SOLVER::GMRES);

	//RelaxationRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	double Time = 8.0;
	double dt = 0.1;
	double collectTime = 8.0;

	arma::Col<double> vel = Solver.setupTestCase(TESTSUITE::SHEAR_LAYER_ROLL_UP);
	arma::Col<double> p = arma::zeros(0.0);

	arma::Col<double> velInit = vel;

	//vel = RK.integrate(Time, dt, vel, p, Solver, collectTime);

	vel.load("final_vel_comp", arma::arma_binary);

	arma::Col<double> velInterp = Solver.interpolateVelocity(vel);

	if (remove("vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	velInterp.save("vel.txt", arma::raw_ascii);

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

	arma::Mat<double> err(1, 60);

	arma::uword i = 0;
	arma::uword j = 0;

	arma::Col<double> velr;

	for (int r = 30; r <= 30; ++r) {

		j = 0;

		for (int m = 1; m <= 60; ++m) {

			int numModesPOD = r;
			int numModesDEIM = m;

			//noHyperReduction hyperReduction;

			//DEIM hyperReduction(numModesDEIM, RK.getDataCollector());

			//SPDEIM hyperReduction(numModesDEIM, RK.getDataCollector());

			LSDEIM hyperReduction(numModesDEIM, m, RK.getDataCollector());

			ROM_Solver RomSolver(Solver, RK.getDataCollector(), numModesPOD, hyperReduction);

			ExplicitRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

			//ImplicitRungeKutta_ROM<false> RKr(ButcherTableaus::implicitMidpoint(), LINEAR_SOLVER::DIRECT);

			//RelaxationRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

			//arma::Col<double> a = RomSolver.calculateIC(vel);

			//arma::Col<double> Nr = hyperReduction.N(a, RomSolver);

			//arma::Col<double> N = Solver.N(RomSolver.Psi() * a);

			//err(i, j) = arma::norm((N - Nr), 2);

			arma::Col<double> a = RomSolver.calculateIC(velInit);
			arma::Col<double> aInit = a;

			a = RKr.integrate(Time, dt, a, p, RomSolver);

			velr = RomSolver.Psi() * a;

			err(i, j) = arma::abs((velr - vel)).max();
			//err(i, j) = arma::norm(velr - vel, 2);

			std::cout << "iteration (" << i << ", " << j << "): " << err(i, j) << std::endl;

			++j;
		}

		++i;
	}

	err.save("error_study.txt", arma::raw_ascii);

	arma::Col<double> RomVelInterp = Solver.interpolateVelocity(velr);

	if (remove("rom_vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	RomVelInterp.save("rom_vel.txt", arma::raw_ascii);

#endif


#ifdef TEST_CODE_DEIM_CONVERGENCE2

	mesh Mesh(100, 100, 2 * PI, 2 * PI); //(400, 400, 1.0, 1.0); 

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

	ExplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	//ImplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::implicitMidpoint(), LINEAR_SOLVER::GMRES);

	//RelaxationRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	double Time = 8.0;
	double dt = 0.01;
	double collectTime = 8.0;

	//arma::Col<double> vel	= arma::linspace(0.0, Mesh.getNumU() + Mesh.getNumV() - 1.0, Mesh.getNumU() + Mesh.getNumV());

	arma::Col<double> vel = Solver.setupTestCase(TESTSUITE::SHEAR_LAYER_ROLL_UP);
	arma::Col<double> p = arma::zeros(0.0);

	arma::Col<double> velInit = vel;

	vel = RK.integrate(Time, dt, vel, p, Solver, collectTime);

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

	arma::uword i = 0;
	
	int numModesPOD = 90;

	int numModesDEIM;

	int numPointsDEIM = 120;

	std::vector<double> errs;

	arma::uword it = 500;

	for (int m = 1; m <= 60; ++m) {

		numModesDEIM = m;

		LSDEIM hyperReduction(numModesDEIM, numPointsDEIM, RK.getDataCollector());

		ROM_Solver RomSolver(Solver, RK.getDataCollector(), numModesPOD, hyperReduction);

		//arma::Col<double> a = RomSolver.calculateIC(velInit);

		arma::Col<double> a = RomSolver.Psi().t() * RomSolver.getSolver().Om() * RK.getDataCollector().getDataMatrix().col(it);

		errs.push_back(arma::norm(RomSolver.getSolver().N(RomSolver.Psi() * a) - hyperReduction.N(a, RomSolver), "inf"));

		std::cout << "deim err: " << errs.back() << std::endl;

		if (m == 1 || m == 60)
			std::cout << "pod err: " << arma::norm(RK.getDataCollector().getDataMatrix().col(it) - RomSolver.Psi() * RomSolver.Psi().t() * RomSolver.getSolver().Om() * RK.getDataCollector().getDataMatrix().col(it), "inf") << std::endl;

		++i;
	}

	std::ofstream outFile("deim_conv.txt");
	for (const auto& err : errs) outFile << err << "\n";

#endif

#ifdef ERROR_ANALYSIS1

	mesh Mesh(256, 256, 2 * PI, 2 * PI); //(400, 400, 1.0, 1.0); 

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

	ExplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	double Time = 8.0;
	double dt = 0.01;
	double collectTime = 8.0;

	arma::Col<double> vel = Solver.setupTestCase(TESTSUITE::SHEAR_LAYER_ROLL_UP);
	arma::Col<double> p = arma::zeros(0.0);

	arma::Col<double> velInit = vel;

	//vel = RK.integrate(Time, dt, vel, p, Solver, collectTime);

	vel.load("final_vel_comp", arma::arma_binary);

	arma::Col<double> velInterp = Solver.interpolateVelocity(vel);

	if (remove("vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	velInterp.save("vel.txt", arma::raw_ascii);

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

	arma::Col<double> velr;

	int r = 30;
	int m = 50;

	int numModesPOD = r;
	int numModesDEIM = m;

	LSDEIM hyperReduction(numModesDEIM, m, RK.getDataCollector());

	ROM_Solver RomSolver(Solver, RK.getDataCollector(), numModesPOD, hyperReduction);

	ExplicitRungeKutta_ROM<true> RKr(ButcherTableaus::RK4());

	arma::Col<double> a = RomSolver.calculateIC(velInit);
	arma::Col<double> aInit = a;

	a = RKr.integrate(Time, dt, a, p, RomSolver, collectTime);

	arma::Mat<double> data = RKr.getDataCollector().getOperatorMatrix();

	data.save("c50_m.txt", arma::raw_ascii);

	//arma::Mat<double> psi;

	//for (int i = 0; i < RomSolver.Psi().n_cols; ++i) {

		//psi.insert_cols(i, Solver.interpolateVelocity(RomSolver.Psi().col(i)));

	//}

	//psi.save("basis.txt", arma::raw_ascii);

	//arma::Mat<double> psim;

	//psim = RomSolver.Psi().t() * hyperReduction.M();

	//psim.save("correlations_DEIMPOD.txt", arma::raw_ascii);

	velr = RomSolver.Psi() * a;

	std::cout << "inf error: " << arma::abs((velr - vel)).max() << std::endl;

	arma::Col<double> RomVelInterp = Solver.interpolateVelocity(velr);

	if (remove("rom_vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	RomVelInterp.save("rom_vel.txt", arma::raw_ascii);

#endif

	return 0;
}
