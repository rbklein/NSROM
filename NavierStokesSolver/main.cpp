#include <iostream>
#include <armadillo>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include <string>
#include <chrono>

#include "data.h"
#include "mesh.h"
#include "solver.h"
#include "boundary.h"
#include "integrators.h"
#include "testsuite.h"

#include "ROM.h"

#include "network.h"
#include "loss.h"
#include "train.h"

//TO DO: mesh does not work for 2x2 (but who cares?)

constexpr double PI = 3.14159265358979323846;
constexpr bool COLLECT_DATA = true;
constexpr bool SAVE_DATA = false;

#define FOM_CODE
//#define ROM_CODE
//#define iROM_CODE
//#define MROM_CODE
//#define TEST_CODE_SPATIAL
//#define TEST_CODE_TEMPORAL
//#define TEST_CODE_DEIM_CONVERGENCE
//#define TEST_CODE_DEIM_CONVERGENCE2

int main() {

	//FULL ORDER MODEL	

#ifdef FOM_CODE

	mesh Mesh(20, 20, 2 * PI, 2 * PI);  //(1024, 1024, 1.0, 1.0); // // (125, 125, 1.0, 1.0); // 
	
	//solver has to check 2x (UL or LR) Periodic is given, otherwise throw error 
	solver Solver(
			Mesh,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			B_CONDITION::PERIODIC_UL,
			B_CONDITION::PERIODIC_LR,
			POISSON_SOLVER::FOURIER,
			0.0 //01
		);

	//ExplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	ImplicitRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::implicitMidpoint(), LINEAR_SOLVER::DIRECT);

	//RelaxationRungeKutta_NS<COLLECT_DATA> RK(ButcherTableaus::RK4());

	double Time = 1.0; //8.0;
	double dt = 0.01; // = 0.0002;
	double collectTime	= 8.0;

	arma::Col<double> vel, velInit, vorticity, streamFunc;
	arma::Col<double> p			= arma::zeros(0.0);
	velInit						= Solver.setupTestCase(TESTSUITE::TAYLOR_GREEN_VORTEX);
	//velInit					= vel;
	//velInit.load("initial_cond_solution_1024_2dturb");

	vel = velInit;

	//arma::Mat<double> data;
	//data.load("solution_snapshots_2dturb_0");
	//vel = data.col(data.n_cols - 1);

	vel = RK.integrate(Time, dt, velInit, p, Solver, collectTime);	

	//RK.getDataCollector().getDataMatrix().save("solution_snapshots_slr_Re100");
	//RK.getDataCollector().getOperatorMatrix().save("operator_snapshots_slr_Re100");

	vorticity = Solver.vorticity(vel);

	streamFunc = Solver.poissonSolve(-vorticity);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	arma::Col<double> velInterp = Solver.interpolateVelocity(vel);

	
	if (remove("vel.txt") == 0)
		std::cout << "removed velocity file" << std::endl;

	velInterp.save("vel.txt", arma::raw_ascii);

	if (remove("vort.txt") == 0)
		std::cout << "removed vorticity file" << std::endl;

	vorticity.save("vort.txt", arma::raw_ascii);

	if (remove("stream.txt") == 0)
		std::cout << "removed stream file" << std::endl;

	streamFunc.save("stream.txt", arma::raw_ascii);
	

	std::cout << "energy: " << 0.5 * arma::as_scalar(vel.t() * Solver.Om() * vel) << std::endl;

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

	/*
	const auto& cells = Solver.getMesh().getCellsU().col(256);

	arma::Col<double> centerlineU(512);

	for (int i = 0; i < 512; ++i) {
		centerlineU(i) = vel(cells(i).vectorIndex);
	}

	const auto& cells2 = Solver.getMesh().getCellsV().row(128);

	arma::Col<double> quarterlineV(512);

	for (int i = 0; i < 512; ++i) {
		quarterlineV(i) = vel(cells2(i).vectorIndex);
	}
	
	centerlineU.save("cetnerlineU512.txt", arma::raw_ascii);
	quarterlineV.save("quaterlineV512.txt", arma::raw_ascii);
	*/

#endif

	//REDUCED ORDER MODEL

#ifdef ROM_CODE

	int numModesPOD = 30;
	int numModesDEIM = 41;  //5, 10, 20, 40, 80
	int numPointsDEIM = 60; // 900; //5, 10, 20, 40, 80

	//noHyperReduction hyperReduction;

	DEIM hyperReduction(numModesDEIM, RK.getDataCollector(), SAVE_DATA);

	//SPDEIM hyperReduction(numModesDEIM, RK.getDataCollector(), SAVE_DATA);

	//LSDEIM hyperReduction(numModesDEIM, numPointsDEIM, RK.getDataCollector(),SAVE_DATA);

	ROM_Solver RomSolver(Solver, RK.getDataCollector(), numModesPOD, hyperReduction, SAVE_DATA);

	ExplicitRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

	//ImplicitRungeKutta_ROM<false> RKr(ButcherTableaus::GL4(), LINEAR_SOLVER::DIRECT);

	//RelaxationRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

	arma::Col<double> a		= RomSolver.calculateIC(velInit);
	arma::Col<double> aInit = a;
	
	auto t1 = std::chrono::high_resolution_clock::now();
	a = RKr.integrate(Time, dt, aInit, p, RomSolver);
	auto t2 = std::chrono::high_resolution_clock::now();

	std::cout << "hROM online time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;


	arma::Col<double> velr	= RomSolver.Psi() * a;

	arma::Col<double> RomVelInterp = Solver.interpolateVelocity(velr);

	if (remove("rom_vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	RomVelInterp.save("rom_vel.txt", arma::raw_ascii);

	arma::Col<double> vorticityr = Solver.vorticity(velr);

	arma::Col<double> streamFuncr = Solver.poissonSolve(-vorticityr);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	if (remove("rom_vort.txt") == 0)
		std::cout << "removed rom vorticity file" << std::endl;

	vorticityr.save("rom_vort.txt", arma::raw_ascii);

	if (remove("rom_stream.txt") == 0)
		std::cout << "removed rom stream file" << std::endl;

	streamFuncr.save("rom_stream.txt", arma::raw_ascii);


	std::cout << "error: " << arma::norm(arma::sqrt(Solver.Om()) * (vel - velr)) << std::endl;

	std::cout << "divergence: " << (Solver.M() * velr).max() << std::endl;

	std::cout << arma::abs((velr - vel)).max() << std::endl;

	std::cout << a.t() * hyperReduction.Nrh(a, RomSolver) << std::endl;

#endif

	//manifold reduced order model

#ifdef MROM_CODE

	ConvolutionalAutoEncoder Network(Solver);

	torch::Device device = getDevice();
	try {
		trainModel(Network, device, 1000, 0.001, RK.getDataCollector().toTensor(RK.getDataCollector().getDataMatrix(), device), 20);
	}
	catch (const c10::Error& e) {
		std::cout << e.msg() << std::endl;
	}
	torch::Tensor input = RK.getDataCollector().toTensor(RK.getDataCollector().getDataMatrix().col(RK.getDataCollector().getDataMatrix().n_cols - 1), device);

	torch::Tensor output = Network->forward(input);

	arma::Col<double> a_output(output.size(0), arma::fill::zeros);

	for (int i = 0; i < output.size(0); ++i) {
		a_output(i) = output.index({ i }).item<double>();
	}

	std::cout << arma::abs(Solver.M() * a_output).max() << std::endl;

	//CHECK RESHAPES

	arma::Col<double> outputInterp = Solver.interpolateVelocity(a_output);

	if (remove("velCAE.txt") == 0)
		std::cout << "removed file" << std::endl;

	outputInterp.save("velCAE.txt", arma::raw_ascii);

	torch::save(Network, "network128.pt");

	std::cout << "done" << std::endl;

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

	arma::Col<double> velInterp = Solver.interpolateVelocity(vel);

	if (remove("vel.txt") == 0)
		std::cout << "removed file" << std::endl;

	velInterp.save("vel.txt", arma::raw_ascii);

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

	arma::Mat<double> err(60 - 2, 60);

	arma::uword i = 0;
	arma::uword j = 0;

	for (int r = 3; r <= 60; ++r) {

		j = 0;

		for (int m = 1; m <= 60; ++m) {

			int numModesPOD = r;
			int numModesDEIM = m;

			//noHyperReduction hyperReduction;

			//DEIM hyperReduction(numModesDEIM, RK.getDataCollector());

			//SPDEIM hyperReduction(numModesDEIM, RK.getDataCollector());

			LSDEIM hyperReduction(numModesDEIM, 120, RK.getDataCollector());

			ROM_Solver RomSolver(Solver, RK.getDataCollector(), numModesPOD, hyperReduction);

			ExplicitRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

			//ImplicitRungeKutta_ROM<false> RKr(ButcherTableaus::GL4(), LINEAR_SOLVER::DIRECT);

			//RelaxationRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

			arma::Col<double> a = RomSolver.calculateIC(velInit);
			arma::Col<double> aInit = a;

			a = RKr.integrate(Time, 0.1 * dt, a, p, RomSolver);

			arma::Col<double> velr = RomSolver.Psi() * a;

			err(i, j) = arma::abs((velr - vel)).max();

			std::cout << "iteration (" << i << ", " << j << "): " << err(i, j) << std::endl;

			++j;
		}

		++i;
	}

	err.save("error_study.txt", arma::raw_ascii);

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

	//interval reduced order model

#ifdef iROM_CODE

	
	int numModesPOD = 37;
	int numModesDEIM = 37;
	int numPointsDEIM = 300;

	LSDEIM hyperReduction8(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 0);
	ROM_Solver RomSolver8(Solver, RK.getDataCollector(), numModesPOD, hyperReduction8, SAVE_DATA, 0);

	arma::Col<double> a = RomSolver8.calculateIC(velInit);
	arma::Col<double> aInit = a;

//#define first_only

#ifndef first_only

	numModesPOD = 19;
	numModesDEIM = 19;
	numPointsDEIM = 100;

	LSDEIM hyperReduction9(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 1);
	ROM_Solver RomSolver9(Solver, RK.getDataCollector(), numModesPOD, hyperReduction9, SAVE_DATA, 1);

	arma::Mat<double> transistionMatrix89 = RomSolver9.Psi().t() * Solver.Om() * RomSolver8.Psi();

	numModesPOD = 12;
	numModesDEIM = 14;
	numPointsDEIM = 100;

	LSDEIM hyperReduction10(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 2);
	ROM_Solver RomSolver10(Solver, RK.getDataCollector(), numModesPOD, hyperReduction10, SAVE_DATA, 2);

	arma::Mat<double> transistionMatrix910 = RomSolver10.Psi().t() * Solver.Om() * RomSolver9.Psi();

	numModesPOD = 12;
	numModesDEIM = 14;
	numPointsDEIM = 100;

	LSDEIM hyperReduction11(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 3);
	ROM_Solver RomSolver11(Solver, RK.getDataCollector(), numModesPOD, hyperReduction11, SAVE_DATA, 3);

	arma::Mat<double> transistionMatrix1011 = RomSolver11.Psi().t() * Solver.Om() * RomSolver10.Psi();

	numModesPOD = 12;
	numModesDEIM = 14;
	numPointsDEIM = 100;

	LSDEIM hyperReduction12(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 4);
	ROM_Solver RomSolver12(Solver, RK.getDataCollector(), numModesPOD, hyperReduction12, SAVE_DATA, 4);

	arma::Mat<double> transistionMatrix1112 = RomSolver12.Psi().t() * Solver.Om() * RomSolver11.Psi();

	numModesPOD = 12;
	numModesDEIM = 14;
	numPointsDEIM = 100;

	LSDEIM hyperReduction13(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 5);
	ROM_Solver RomSolver13(Solver, RK.getDataCollector(), numModesPOD, hyperReduction13, SAVE_DATA, 5);

	arma::Mat<double> transistionMatrix1213 = RomSolver13.Psi().t() * Solver.Om() * RomSolver12.Psi();

	numModesPOD = 12;
	numModesDEIM = 14;
	numPointsDEIM = 100;
	
	LSDEIM hyperReduction14(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 6);
	ROM_Solver RomSolver14(Solver, RK.getDataCollector(), numModesPOD, hyperReduction14, SAVE_DATA, 6);

	arma::Mat<double> transistionMatrix1314 = RomSolver14.Psi().t() * Solver.Om() * RomSolver13.Psi();
#endif
	/*

	LSDEIM hyperReduction15(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 15);
	ROM_Solver RomSolver15(Solver, RK.getDataCollector(), numModesPOD, hyperReduction15, SAVE_DATA, 15);

	arma::Mat<double> transistionMatrix1415 = RomSolver15.Psi().t() * Solver.Om() * RomSolver14.Psi();

	LSDEIM hyperReduction16(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 16);
	ROM_Solver RomSolver16(Solver, RK.getDataCollector(), numModesPOD, hyperReduction16, SAVE_DATA, 16);

	arma::Mat<double> transistionMatrix1516 = RomSolver16.Psi().t() * Solver.Om() * RomSolver15.Psi();

	LSDEIM hyperReduction17(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 17);
	ROM_Solver RomSolver17(Solver, RK.getDataCollector(), numModesPOD, hyperReduction17, SAVE_DATA, 17);

	arma::Mat<double> transistionMatrix1617 = RomSolver17.Psi().t() * Solver.Om() * RomSolver16.Psi();

	LSDEIM hyperReduction18(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 18);
	ROM_Solver RomSolver18(Solver, RK.getDataCollector(), numModesPOD, hyperReduction18, SAVE_DATA, 18);

	arma::Mat<double> transistionMatrix1718 = RomSolver18.Psi().t() * Solver.Om() * RomSolver17.Psi();

	LSDEIM hyperReduction19(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 19);
	ROM_Solver RomSolver19(Solver, RK.getDataCollector(), numModesPOD, hyperReduction19, SAVE_DATA, 19);

	arma::Mat<double> transistionMatrix1819 = RomSolver19.Psi().t() * Solver.Om() * RomSolver18.Psi();

	LSDEIM hyperReduction20(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 20);
	ROM_Solver RomSolver20(Solver, RK.getDataCollector(), numModesPOD, hyperReduction20, SAVE_DATA, 20);

	arma::Mat<double> transistionMatrix1920 = RomSolver20.Psi().t() * Solver.Om() * RomSolver19.Psi();

	LSDEIM hyperReduction21(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 21);
	ROM_Solver RomSolver21(Solver, RK.getDataCollector(), numModesPOD, hyperReduction21, SAVE_DATA, 21);

	arma::Mat<double> transistionMatrix2021 = RomSolver21.Psi().t() * Solver.Om() * RomSolver20.Psi();

	LSDEIM hyperReduction22(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 22);
	ROM_Solver RomSolver22(Solver, RK.getDataCollector(), numModesPOD, hyperReduction22, SAVE_DATA, 22);

	arma::Mat<double> transistionMatrix2122 = RomSolver22.Psi().t() * Solver.Om() * RomSolver21.Psi();

	LSDEIM hyperReduction23(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 23);
	ROM_Solver RomSolver23(Solver, RK.getDataCollector(), numModesPOD, hyperReduction23, SAVE_DATA, 23);

	arma::Mat<double> transistionMatrix2223 = RomSolver23.Psi().t() * Solver.Om() * RomSolver22.Psi();

	LSDEIM hyperReduction24(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 24);
	ROM_Solver RomSolver24(Solver, RK.getDataCollector(), numModesPOD, hyperReduction24, SAVE_DATA, 24);

	arma::Mat<double> transistionMatrix2324 = RomSolver24.Psi().t() * Solver.Om() * RomSolver23.Psi();

	LSDEIM hyperReduction25(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 25);
	ROM_Solver RomSolver25(Solver, RK.getDataCollector(), numModesPOD, hyperReduction25, SAVE_DATA, 25);

	arma::Mat<double> transistionMatrix2425 = RomSolver25.Psi().t() * Solver.Om() * RomSolver24.Psi();

	LSDEIM hyperReduction26(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 26);
	ROM_Solver RomSolver26(Solver, RK.getDataCollector(), numModesPOD, hyperReduction26, SAVE_DATA, 26);

	arma::Mat<double> transistionMatrix2526 = RomSolver26.Psi().t() * Solver.Om() * RomSolver25.Psi();

	LSDEIM hyperReduction27(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 27);
	ROM_Solver RomSolver27(Solver, RK.getDataCollector(), numModesPOD, hyperReduction27, SAVE_DATA, 27);

	arma::Mat<double> transistionMatrix2627 = RomSolver27.Psi().t() * Solver.Om() * RomSolver26.Psi();

	LSDEIM hyperReduction28(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 28);
	ROM_Solver RomSolver28(Solver, RK.getDataCollector(), numModesPOD, hyperReduction28, SAVE_DATA, 28);

	arma::Mat<double> transistionMatrix2728 = RomSolver28.Psi().t() * Solver.Om() * RomSolver27.Psi();

	LSDEIM hyperReduction29(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 29);
	ROM_Solver RomSolver29(Solver, RK.getDataCollector(), numModesPOD, hyperReduction29, SAVE_DATA, 29);

	arma::Mat<double> transistionMatrix2829 = RomSolver29.Psi().t() * Solver.Om() * RomSolver28.Psi();

	LSDEIM hyperReduction30(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 30);
	ROM_Solver RomSolver30(Solver, RK.getDataCollector(), numModesPOD, hyperReduction30, SAVE_DATA, 30);

	arma::Mat<double> transistionMatrix2930 = RomSolver30.Psi().t() * Solver.Om() * RomSolver29.Psi();

	LSDEIM hyperReduction31(numModesDEIM, numPointsDEIM, RK.getDataCollector(), SAVE_DATA, 31);
	ROM_Solver RomSolver31(Solver, RK.getDataCollector(), numModesPOD, hyperReduction31, SAVE_DATA, 31);

	arma::Mat<double> transistionMatrix3031 = RomSolver31.Psi().t() * Solver.Om() * RomSolver30.Psi();
	*/

	ExplicitRungeKutta_ROM<false> RKr(ButcherTableaus::RK4());

	Time = 0.5;

	a = RKr.integrate(Time - dt, dt, a, p, RomSolver8);
	
	arma::Col<double> velr = RomSolver8.Psi() * a;

	arma::Col<double> vorticityr = Solver.vorticity(velr);

	arma::Col<double> streamFuncr = Solver.poissonSolve(-vorticityr);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	arma::Col<double> velInterpr = Solver.interpolateVelocity(velr);

	velInterpr.save("rom_vel_0.txt", arma::raw_ascii);

	vorticityr.save("rom_vort_0.txt", arma::raw_ascii);

	streamFuncr.save("rom_stream_0.txt", arma::raw_ascii);

#ifndef first_only
	a = transistionMatrix89 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver9);

	velr = RomSolver9.Psi() * a;

	vorticityr = Solver.vorticity(velr);

	streamFuncr = Solver.poissonSolve(-vorticityr);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	velInterpr = Solver.interpolateVelocity(velr);

	velInterpr.save("rom_vel_1.txt", arma::raw_ascii);

	vorticityr.save("rom_vort_1.txt", arma::raw_ascii);

	streamFuncr.save("rom_stream_1.txt", arma::raw_ascii);

	a = transistionMatrix910 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver10);

	a = transistionMatrix1011 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver11);

	velr = RomSolver11.Psi() * a;

	vorticityr = Solver.vorticity(velr);

	streamFuncr = Solver.poissonSolve(-vorticityr);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	velInterpr = Solver.interpolateVelocity(velr);

	velInterpr.save("rom_vel_2.txt", arma::raw_ascii);

	vorticityr.save("rom_vort_2.txt", arma::raw_ascii);

	streamFuncr.save("rom_stream_2.txt", arma::raw_ascii);

	a = transistionMatrix1112 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver12);
	
	a = transistionMatrix1213 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver13);

	a = transistionMatrix1314 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver14);

	velr = RomSolver14.Psi() * a;

	vorticityr = Solver.vorticity(velr);

	streamFuncr = Solver.poissonSolve(-vorticityr);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	velInterpr = Solver.interpolateVelocity(velr);

	velInterpr.save("rom_vel_3.txt", arma::raw_ascii);

	vorticityr.save("rom_vort_3.txt", arma::raw_ascii);

	streamFuncr.save("rom_stream_3.txt", arma::raw_ascii);

	//arma::Col<double> velr = RomSolver14.Psi() * a;

	/*

	a = transistionMatrix1415 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver15);

	a = transistionMatrix1516 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver16);

	a = transistionMatrix1617 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver17);

	a = transistionMatrix1718 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver18);

	a = transistionMatrix1819 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver19);

	a = transistionMatrix1920 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver20);

	a = transistionMatrix2021 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver21);

	a = transistionMatrix2122 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver22);

	a = transistionMatrix2223 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver23);

	a = transistionMatrix2324 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver24);

	a = transistionMatrix2425 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver25);

	a = transistionMatrix2526 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver26);

	a = transistionMatrix2627 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver27);

	a = transistionMatrix2728 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver28);

	a = transistionMatrix2829 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver29);

	a = transistionMatrix2930 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver30);

	a = transistionMatrix3031 * a;

	a = RKr.integrate(Time, dt, a, p, RomSolver31);

	arma::Col<double> velr = RomSolver31.Psi() * a;
	*/


	/*
	arma::Col<double> vorticityr = Solver.vorticity(velr);

	arma::Col<double> streamFuncr = Solver.poissonSolve(-vorticityr);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	arma::Col<double> velInterpr = Solver.interpolateVelocity(velr);
	
	if (remove("rom_vel.txt") == 0)
		std::cout << "removed rom velocity file" << std::endl;

	velInterpr.save("rom_vel.txt", arma::raw_ascii);

	if (remove("rom_vort.txt") == 0)
		std::cout << "removed rom vorticity file" << std::endl;

	vorticityr.save("rom_vort.txt", arma::raw_ascii);

	if (remove("rom_stream.txt") == 0)
		std::cout << "removed rom stream file" << std::endl;

	streamFuncr.save("rom_stream.txt", arma::raw_ascii);
	*/
#endif
	std::cout << "rom energy: " << 0.5 * arma::as_scalar(velr.t() * Solver.Om() * velr) << std::endl;

	std::cout << "rom divergence: " << (Solver.M() * velr).max() << std::endl;

#endif


	return 0;
}


/*
arma::Mat<double> data;
data.load("operator_snapshots_2dturb_31");

arma::Col<double> vec = data.col(0);

//std::cout << "diff deim: " << arma::abs(vec - hyperReduction.M() * arma::solve(hyperReduction.P().t() * hyperReduction.M(), hyperReduction.P().t() * vec)).max() << std::endl;

data.clear();

data.load("solution_snapshots_2dturb_31");

vec = data.col(0);

vec = Solver.Om() * vec;

arma::Col<double> vec2 = RomSolver.Psi() * RomSolver.Psi().t() * vec;

std::cout << "diff pod: " << arma::norm(vec - vec2) << std::endl;

arma::Col<double> RomVelInterp = Solver.interpolateVelocity(vec2);

if (remove("rom_vel.txt") == 0)
	std::cout << "removed file" << std::endl;

RomVelInterp.save("rom_vel.txt", arma::raw_ascii);
*/


/*

mesh Mesh(450, 450, 1.0, 1.0); // (128, 128, 2 * PI, 2 * PI);

	//solver has to check 2x (UL or LR) Periodic is given, otherwise throw error
	solver Solver(
		Mesh,
		B_CONDITION::PERIODIC_UL,
		B_CONDITION::PERIODIC_LR,
		B_CONDITION::PERIODIC_UL,
		B_CONDITION::PERIODIC_LR,
		POISSON_SOLVER::FOURIER,
		0.00001
	);

	const arma::field<cell>& CellsU = Solver.getMesh().getCellsU();
	const arma::field<cell>& CellsV = Solver.getMesh().getCellsV();

	arma::Col<double> Eu = arma::zeros(Solver.getMesh().getNumU() + Solver.getMesh().getNumV());
	arma::Col<double> Ev = arma::zeros(Solver.getMesh().getNumU() + Solver.getMesh().getNumV());

	//setup momentum conserving modes
	for (arma::uword i = Solver.getMesh().getStartIndUy(); i < Solver.getMesh().getEndIndUy(); ++i) {
		for (arma::uword j = Solver.getMesh().getStartIndUx(); j < Solver.getMesh().getEndIndUx(); ++j) {
			Eu(CellsU(i, j).vectorIndex) = 1.0;
		}
	}

	for (arma::uword i = Solver.getMesh().getStartIndVy(); i < Solver.getMesh().getEndIndVy(); ++i) {
		for (arma::uword j = Solver.getMesh().getStartIndVx(); j < Solver.getMesh().getEndIndVx(); ++j) {
			Ev(CellsV(i, j).vectorIndex) = 1.0;
		}
	}

	//normalize momentum conserving modes in omega-norm
	Eu = (1.0 / sqrt(arma::as_scalar(Eu.t() * Solver.Om() * Eu))) * Eu;
	Ev = (1.0 / sqrt(arma::as_scalar(Ev.t() * Solver.Om() * Ev))) * Ev;

	arma::Mat<double> E = arma::join_rows(Eu, Ev);


	// PRE SVD

	arma::Mat<double> scaledSnapshotData;

	arma::Col<double> ee;

	for (int i = 0; i < 32; ++i) {

		std::cout << i << std::endl;

		//get snapshot data
		scaledSnapshotData.load("solution_snapshots_2dturb_" + std::to_string(i));

		for (int j = 0; j < scaledSnapshotData.n_cols; ++j) {

			ee = E.t() * Solver.Om() * scaledSnapshotData.col(j);

			//subtract omega-weighted projections of snapshots on E
			scaledSnapshotData.col(j) = scaledSnapshotData.col(j) - E * ee;
		}

		//scale snapshots for omega-orthogonality
		scaledSnapshotData = arma::sqrt(Solver.Om()) * scaledSnapshotData;

		scaledSnapshotData.save("solution_scaled_snapshots_2dturb_" + std::to_string(i));

		scaledSnapshotData.clear();
	}

*/




/*
data.load("solution_snapshots_2dturb_31");

arma::Col<double> v = data.col(data.n_cols - 1);
arma::Col<double> acheck = RomSolver.Psi().t() * Solver.Om() * v;

std::cout << arma::abs(v - RomSolver.Psi() * acheck).max() << std::endl;

arma::Col<double> velr = Solver.interpolateVelocity(RomSolver.Psi() * acheck);

velr.save("rom_vel.txt", arma::raw_ascii);

data.clear();

data.load("operator_snapshots_2dturb_31");

v = data.col(data.n_cols - 1);

acheck = hyperReduction.M().t() * v;

std::cout << arma::abs(v - hyperReduction.M() * acheck).max() << std::endl;

velr = Solver.interpolateVelocity(hyperReduction.M() * acheck);

velr.save("rom_operator.txt", arma::raw_ascii);

*/




/*
	arma::Mat<double> rSingVecsX;
	arma::Col<double> singValsX;

	arma::Mat<double> rSingVecsXi;
	arma::Col<double> singValsXi;

	rSingVecsX.load("right_sing_vecs_solution");
	rSingVecsXi.load("right_sing_vecs_operator");

	singValsX.load("sing_vals_solution");
	singValsXi.load("sing_vals_operator");

	int r = 1500;

	int layerThickness = 67500;
	int numLayers = 6;
	int N = 405000;

	arma::Mat<double> DEIMmodes;

	arma::Mat<double> placeholder1;
	arma::Mat<double> placeholder2;

	for (int i = 0; i < numLayers; ++i) {

		for (int j = 0; j < 32; ++j) {

			placeholder1.load("operator_snapshots_2dturb_" + std::to_string(j));

			placeholder2 = arma::join_rows(placeholder2, placeholder1.rows(i * layerThickness, (i + 1) * layerThickness - 1));

			placeholder1.clear();

			std::cout << i << " " << j << std::endl;

		}

		DEIMmodes = arma::join_cols(DEIMmodes, placeholder2 * rSingVecsXi.cols(0, r - 1));

		placeholder2.clear();
	}

	DEIMmodes.save("deim_modes_intermediate");

	DEIMmodes = DEIMmodes * arma::inv(arma::diagmat(singValsXi.subvec(0, r - 1)));

	std::cout << DEIMmodes.n_rows << " " << DEIMmodes.n_cols << std::endl;

	DEIMmodes.save("deim_modes_1500.txt", arma::raw_ascii);
	*/

/*
	arma::Mat<double> rSingVecsX;
	arma::Col<double> singValsX;

	arma::Mat<double> rSingVecsXi;
	arma::Col<double> singValsXi;

	rSingVecsX.load("right_sing_vecs_solution");
	rSingVecsXi.load("right_sing_vecs_operator");

	singValsX.load("sing_vals_solution");
	singValsXi.load("sing_vals_operator");

	int r = 300;

	int layerThickness = 67500;
	int numLayers = 6;
	int N = 405000;

	arma::Mat<double> placeholder1;
	arma::Mat<double> placeholder2;

	arma::Mat<double> PODmodes;

	for (int i = 0; i < numLayers; ++i) {

		for (int j = 0; j < 32; ++j) {

			placeholder1.load("solution_snapshots_2dturb_" + std::to_string(j));

			placeholder2 = arma::join_rows(placeholder2, placeholder1.rows(i * layerThickness, (i + 1) * layerThickness - 1));

			placeholder1.clear();

			std::cout << i << " " << j << std::endl;

		}

		PODmodes = arma::join_cols(PODmodes, placeholder2 * rSingVecsX.cols(0, r - 1));

		placeholder2.clear();
	}

	PODmodes.save("pod_modes_intermediate");

	PODmodes = PODmodes * arma::inv(arma::diagmat(singValsX.subvec(0, r - 1)));

	PODmodes.save("pod_modes_300");

*/



/*
	arma::Mat<double> eigvecX;
	arma::Col<double> eigvalX;
	arma::Mat<double> eigvecXi;
	arma::Col<double> eigvalXi;

	arma::Mat<double> XTX;
	arma::Mat<double> XiTXi;

	XTX.load("correlation_matrix_snapshots");
	XiTXi.load("correlation_matrix_operator");

	arma::eig_sym(eigvalX, eigvecX, XTX);
	arma::eig_sym(eigvalXi, eigvecXi, XiTXi);

	eigvecX = arma::fliplr(eigvecX);
	eigvecXi = arma::fliplr(eigvecXi);

	eigvecX.save("right_sing_vecs_solution");
	eigvecXi.save("right_sing_vecs_operator");

	arma::Col<double> singValsX = arma::sqrt(eigvalX);
	singValsX = arma::reverse(singValsX);
	singValsX.save("sing_vals_solution");

	arma::Col<double> singValsXi = arma::sqrt(eigvalXi);
	singValsXi = arma::reverse(singValsXi);
	singValsXi.save("sing_vals_operator");

	singValsX.save("sing_vals_solution1.txt", arma::raw_ascii);
	singValsXi.save("sing_vals_operator1.txt", arma::raw_ascii);
	*/




/*
	arma::Mat<double> XTX;
	arma::Mat<double> placeholder1;
	arma::Mat<double> placeholder2;
	arma::Mat<double> placeholder3;

	for (int i = 0; i < 32; ++i) {

		placeholder1.load("solution_snapshots_2dturb_" + std::to_string(i));

		for (int j = 0; j < 32; ++j) {

			placeholder2.load("solution_snapshots_2dturb_" + std::to_string(j));

			placeholder3 = arma::join_rows(placeholder3, placeholder1.t() * placeholder2);

			placeholder2.clear();

			std::cout << i << " " << j << std::endl;
		}

		XTX = arma::join_cols(XTX, placeholder3);

		placeholder1.clear();
		placeholder3.clear();

	}

	std::cout << XTX.n_rows << " " << XTX.n_cols << std::endl;

	XTX.save("correlation_matrix_snapshots");


	XTX.clear();
	placeholder1.clear();
	placeholder2.clear();
	placeholder3.clear();

	for (int i = 0; i < 32; ++i) {

		placeholder1.load("operator_snapshots_2dturb_" + std::to_string(i));

		for (int j = 0; j < 32; ++j) {

			placeholder2.load("operator_snapshots_2dturb_" + std::to_string(j));

			placeholder3 = arma::join_rows(placeholder3, placeholder1.t() * placeholder2);

			placeholder2.clear();

			std::cout << i << " " << j << std::endl;
		}

		XTX = arma::join_cols(XTX, placeholder3);

		placeholder1.clear();
		placeholder3.clear();

	}

	std::cout << XTX.n_rows << " " << XTX.n_cols << std::endl;

	XTX.save("correlation_matrix_operator");
*/

