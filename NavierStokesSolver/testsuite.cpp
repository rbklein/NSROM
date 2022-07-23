#include <iostream>
#include <armadillo>
#include <cmath>
#include <thread>
#include <functional>
#include <random>
#include <chrono>

#include "mesh.h"
#include "solver.h"
#include "testsuite.h"

constexpr int NUM_THREADS = 8;

constexpr double PI = 3.14159265358979323846;

arma::Col<double> solver::setupTestCase(TESTSUITE testcase) const {

	arma::Col<double> V = arma::zeros(m_mesh.getNumU() + m_mesh.getNumV());

	const arma::field<cell>& CellsU = m_mesh.getCellsU();
	const arma::field<cell>& CellsV = m_mesh.getCellsV();
	const arma::field<cell>& CellsP = m_mesh.getCellsP();

	switch (testcase) {
	case(TESTSUITE::TAYLOR_GREEN_VORTEX):
	{
		for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

				V(CellsU(i, j).vectorIndex) = cos(CellsU(i, j).x) * sin(CellsU(i, j).y);

			}
		}

		for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

				V(CellsV(i, j).vectorIndex) = -1.0 * sin(CellsV(i, j).x) * cos(CellsV(i, j).y);

			}
		}
	}
		break;

	case(TESTSUITE::SHEAR_LAYER_ROLL_UP):
	{
		double delta = PI / 15.0;
		double epsilon = 0.05;

		for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

				if (CellsU(i, j).y <= PI) {
					V(CellsU(i, j).vectorIndex) = tanh((CellsU(i, j).y - PI / 2.0) / delta);
				}
				else {
					V(CellsU(i, j).vectorIndex) = tanh((3.0 * PI / 2.0 - CellsU(i, j).y) / delta);
				}

			}
		}

		for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

				V(CellsV(i, j).vectorIndex) = epsilon * sin(CellsV(i, j).x);

			}
		}
	}
		break;

	case(TESTSUITE::FREELY_DECAYING_2D_TURBULENCE):
	{
		//number of initial vortices in each coordinate direction
		int nv = 16;

		//stream function associated to initial condition
		arma::Col<double> streamFunc(m_mesh.getNumCellsX() * m_mesh.getNumCellsY());

		arma::SpMat<double> zerosU(m_mesh.getNumU(), m_mesh.getNumU());
		arma::SpMat<double> zerosV(m_mesh.getNumV(), m_mesh.getNumV());

		arma::SpMat<double> diag1 = arma::speye(m_mesh.getNumU(), m_mesh.getNumU());
		arma::SpMat<double> diagm1 = -1.0 * arma::speye(m_mesh.getNumV(), m_mesh.getNumV());

		arma::SpMat<double> matrix = arma::join_cols(arma::join_rows(zerosU, diag1), arma::join_rows(diagm1, zerosV));

		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(0.0, 0.01);

		std::vector<double> dx;
		std::vector<double> dy;

		for (int i = 0; i < (16 * 16); ++i) {
			dx.push_back(distribution(generator));
			dy.push_back(distribution(generator));
		}

		//dx.save("initial_offsets_x");
		//dy.save("initial_offsets_y");
	
		//dx.load("initial_offsets_x");
		//dy.load("initial_offsets_y");

		int count = 0;

		double offsetX = CellsP(0, 0).dx / 2.0;
		double offsetY = CellsP(0, 0).dy / 2.0;

		//for all p unknowns
		for (arma::uword i = 0; i < m_mesh.getNumCellsY(); ++i) {
			for (arma::uword j = 0; j < m_mesh.getNumCellsX(); ++j) {

				streamFunc(CellsP(i, j).vectorIndex) = 0.0;

				for (int q = 2; q < (nv + 2); ++q) {
					for (int p = 2; p < (nv + 2); ++p) {

						//try to add some randomness in vortex strength or something

						//careful: some implicit casting going on in denominator
						streamFunc(CellsP(i, j).vectorIndex) += 0.05 * std::pow(-1.0, q + p) * exp(-2000.0 *
							(std::pow(CellsP(i, j).x - offsetX + dx[count] - p / (nv + 3.0), 2.0) + std::pow(CellsP(i, j).y - offsetY + dy[count] - q / (nv + 3.0), 2.0))
						);


						++count;

					}
				}

				count = 0;

			}
		}

		//V = matrix * m_OmegaInv * m_G * streamFunc;

		V = curlStream(streamFunc);

	}
		break;


	case(TESTSUITE::VORTEX_MERGER):
	{

		//stream function associated to initial condition
		arma::Col<double> streamFunc(m_mesh.getNumCellsX() * m_mesh.getNumCellsY());

		arma::SpMat<double> zerosU(m_mesh.getNumU(), m_mesh.getNumU());
		arma::SpMat<double> zerosV(m_mesh.getNumV(), m_mesh.getNumV());

		arma::SpMat<double> diag1 = arma::speye(m_mesh.getNumU(), m_mesh.getNumU());
		arma::SpMat<double> diagm1 = -1.0 * arma::speye(m_mesh.getNumV(), m_mesh.getNumV());

		arma::SpMat<double> matrix = arma::join_cols(arma::join_rows(zerosU, diag1), arma::join_rows(diagm1, zerosV));

		//for all p unknowns
		for (arma::uword i = 0; i < m_mesh.getNumCellsY(); ++i) {
			for (arma::uword j = 0; j < m_mesh.getNumCellsX(); ++j) {

				streamFunc(CellsP(i, j).vectorIndex) = 0.0;

				streamFunc(CellsP(i, j).vectorIndex) += 0.1 * exp(-100.0 *
					(std::pow(CellsP(i, j).x - 0.6, 2.0) + std::pow(CellsP(i, j).y - 0.4, 2.0))
				);

				streamFunc(CellsP(i, j).vectorIndex) += 0.1 * exp(-100.0 *
					(std::pow(CellsP(i, j).x - 0.4, 2.0) + std::pow(CellsP(i, j).y - 0.6, 2.0))
				);

			}
		}

		V = matrix * m_OmegaInv * m_G * streamFunc;
	}
	break;

	case(TESTSUITE::SHEAR_LAYER_ROLL_UP2):
	{
		double delta = PI / 1500.0;
		double epsilon = 0.5;

		for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

				if (fmod(CellsU(i, j).y, PI) <= (0.5 * PI)) {
					V(CellsU(i, j).vectorIndex) = tanh((fmod(CellsU(i, j).y, PI) - PI / 4.0) / delta);
				}
				else {
					V(CellsU(i, j).vectorIndex) = tanh((3.0 * PI / 4.0 - fmod(CellsU(i, j).y, PI)) / delta);
				}

			}
		}

		for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

				V(CellsV(i, j).vectorIndex) = epsilon * sin(CellsV(i, j).x);

			}
		}
	}
	break;

	case(TESTSUITE::TRIANGULAR_LATTICE):
	{
		for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

				V(CellsU(i, j).vectorIndex) = cos(CellsU(i, j).y) * cos(sqrt(3) * CellsU(i,j).x) + cos(CellsU(i,j).y) * cos(CellsU(i, j).y) - sin(CellsU(i, j).y) * sin(CellsU(i, j).y);

			}
		}

		for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

				V(CellsV(i, j).vectorIndex) = sqrt(3) * sin(CellsV(i, j).y) * sin(sqrt(3) * CellsV(i, j).x);

			}
		}
	}
	break;


	}


	return V;
}

/*
for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
				for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

					V(CellsU(i, j).vectorIndex) = 0.0;

					for (int q = 1; q < (nv + 1); ++q) {
						for (int p = 1; p < (nv + 1); ++p) {

							//careful: some implicit casting going on in denominator
							V(CellsU(i, j).vectorIndex) += 0.01 * std::pow(-1.0, q + p) * 2.0 * (CellsU(i, j).y - q / (nv + 1.0)) * exp(-10000.0 *
								(std::pow(CellsU(i, j).x - p / (nv + 1.0), 2.0) + std::pow(CellsU(i, j).y - q / (nv + 1.0), 2.0))
							);

						}
					}

				}
			}
		}

		for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

				V(CellsV(i, j).vectorIndex) = 0.0;

				for (int q = 1; q < (nv + 1); ++q) {
					for (int p = 1; p < (nv + 1); ++p) {

						//careful: some implicit casting going on in denominator
						V(CellsV(i, j).vectorIndex) += 0.01 * std::pow(-1.0, q + p) * (-2.0 * (CellsV(i, j).x - p / (nv + 1.0))) * exp(-10000.0 *
							(std::pow(CellsV(i, j).x - p / (nv + 1.0), 2.0) + std::pow(CellsV(i, j).y - q / (nv + 1.0), 2.0))
						);

					}
				}

			}
		}
*/



/*
		//stream function associated to initial condition
		arma::Col<double> streamFunc(m_mesh.getNumCellsX() * m_mesh.getNumCellsY());

		arma::SpMat<double> zerosU(m_mesh.getNumU(), m_mesh.getNumU());
		arma::SpMat<double> zerosV(m_mesh.getNumV(), m_mesh.getNumV());

		arma::SpMat<double> diag1 = arma::speye(m_mesh.getNumU(), m_mesh.getNumU());
		arma::SpMat<double> diagm1 = -1.0 * arma::speye(m_mesh.getNumV(), m_mesh.getNumV());

		arma::SpMat<double> matrix = arma::join_cols(arma::join_rows(zerosU, diag1), arma::join_rows(diagm1, zerosV));

		//for all p unknowns
		for (arma::uword i = 0; i < m_mesh.getNumCellsY(); ++i) {
			for (arma::uword j = 0; j < m_mesh.getNumCellsX(); ++j) {

				streamFunc(CellsP(i, j).vectorIndex) = 0.0;

				for (int q = 1; q < (nv + 1); ++q) {
					for (int p = 1; p < (nv + 1); ++p) {

						//careful: some implicit casting going on in denominator
						streamFunc(CellsP(i, j).vectorIndex) += 0.01 * std::pow(-1.0, q + p) * exp(-10000.0 *
							(std::pow(CellsP(i, j).x - p / (nv + 1.0), 2.0) + std::pow(CellsP(i, j).y - q / (nv + 1.0), 2.0))
						);

					}
				}

			}
		}

		V = matrix * m_OmegaInv * m_G * streamFunc;
		*/

/*
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(0.0, 0.01);

		std::vector<double> dx;
		std::vector<double> dy;

		for (int i = 0; i < (16 * 16); ++i) {
			dx.push_back(distribution(generator));
			dy.push_back(distribution(generator));
		}

		int count = 0;

		for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

				V(CellsU(i, j).vectorIndex) = 0.0;

				for (int q = 1; q < (nv + 1); ++q) {
					for (int p = 1; p < (nv + 1); ++p) {

						//try to add some randomness in vortex strength or something

						//careful: some implicit casting going on in denominator
						V(CellsU(i, j).vectorIndex) += - 0.05 * std::pow(-1.0, q + p) * -2000.0 * 2.0 * (CellsU(i, j).y + dy[count] - q / (nv + 1.0)) * exp(-2000.0 *
							(std::pow(CellsU(i, j).x + dx[count] - p / (nv + 1.0), 2.0) + std::pow(CellsU(i, j).y + dy[count] - q / (nv + 1.0), 2.0))
						);


						++count;

					}
				}

				count = 0;

			}
		}

		for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
			for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

				V(CellsV(i, j).vectorIndex) = 0.0;

				for (int q = 1; q < (nv + 1); ++q) {
					for (int p = 1; p < (nv + 1); ++p) {

						//try to add some randomness in vortex strength or something

						//careful: some implicit casting going on in denominator
						V(CellsV(i, j).vectorIndex) += 0.05 * std::pow(-1.0, q + p) * -2000.0 * 2.0 * (CellsV(i, j).x + dx[count] - p / (nv + 1.0)) * exp(-2000.0 *
							(std::pow(CellsV(i, j).x + dx[count] - p / (nv + 1.0), 2.0) + std::pow(CellsV(i, j).y + dy[count] - q / (nv + 1.0), 2.0))
						);


						++count;

					}
				}

				count = 0;

			}
		}

*/