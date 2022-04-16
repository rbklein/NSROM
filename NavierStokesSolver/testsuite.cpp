#include <iostream>
#include <armadillo>
#include <cmath>

#include "mesh.h"
#include "solver.h"
#include "testsuite.h"

constexpr double PI = 3.14159265358979323846;

arma::Col<double> solver::setupTestCase(TESTSUITE testcase) {

	arma::Col<double> V = arma::zeros(m_mesh.getNumU() + m_mesh.getNumV());

	const arma::field<cell>& CellsU = m_mesh.getCellsU();
	const arma::field<cell>& CellsV = m_mesh.getCellsV();

	switch (testcase) {
	case(TESTSUITE::TAYLOR_GREEN_VORTEX):

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

		break;

	case(TESTSUITE::SHEAR_LAYER_ROLL_UP):

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

		break;
	}

	return V;
}