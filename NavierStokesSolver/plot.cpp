#include <iostream>
#include <armadillo>
#include <string>

#include "solver.h"
#include "plot.h"

void plot(const solver& Solver, const arma::Col<double>& vel, bool is_rom) {

	arma::Col<double> velInit, vorticity, streamFunc;

	vorticity = Solver.vorticity(vel);

	streamFunc = Solver.poissonSolve(-vorticity);  //streamFunc must be multiplied with dx * dy for correct value (compensates for integration over "pressure" cell)

	arma::Col<double> velInterp = Solver.interpolateVelocity(vel);

	if (is_rom) {

		if (remove("rom_vel.txt") == 0)
			std::cout << "removed velocity file" << std::endl;

		velInterp.save("rom_vel.txt", arma::raw_ascii);

		if (remove("rom_vort.txt") == 0)
			std::cout << "removed vorticity file" << std::endl;

		vorticity.save("rom_vort.txt", arma::raw_ascii);

		if (remove("rom_stream.txt") == 0)
			std::cout << "removed stream file" << std::endl;

		streamFunc.save("rom_stream.txt", arma::raw_ascii);

	}
	else {
		
		if (remove("vel.txt") == 0)
			std::cout << "removed velocity file" << std::endl;

		velInterp.save("vel.txt", arma::raw_ascii);

		if (remove("vort.txt") == 0)
			std::cout << "removed vorticity file" << std::endl;

		vorticity.save("vort.txt", arma::raw_ascii);

		if (remove("stream.txt") == 0)
			std::cout << "removed stream file" << std::endl;

		streamFunc.save("stream.txt", arma::raw_ascii);

	}

	std::cout << "energy: " << 0.5 * arma::as_scalar(vel.t() * Solver.Om() * vel) << std::endl;

	std::cout << "divergence: " << (Solver.M() * vel).max() << std::endl;

}