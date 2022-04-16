#include <iostream>
#include <armadillo>
#include <utility>
#include <vector>

#include "mesh.h"
#include "solver.h"

std::pair<arma::SpMat<double>, arma::SpMat<double>> solver::setupOmegaMatrices() {

	const arma::field<cell>& CellsU = m_mesh.getCellsU();
	const arma::field<cell>& CellsV = m_mesh.getCellsV();

	std::vector<arma::uword> rowIndices;
	std::vector<arma::uword> columnIndices;
	std::vector<double> values;
	std::vector<double> valuesInv;

	arma::uword eqCounter = 0;

	//for all U unknowns
	for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
		for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

			rowIndices.push_back(eqCounter);
			columnIndices.push_back(eqCounter);

			values.push_back(CellsU(i, j).dx * CellsU(i, j).dy);

			valuesInv.push_back(1.0 / (CellsU(i, j).dx * CellsU(i, j).dy));

			++eqCounter;
		}
	}

	//for all V unknowns
	for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
		for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

			rowIndices.push_back(eqCounter);
			columnIndices.push_back(eqCounter);

			values.push_back(CellsV(i, j).dx * CellsV(i, j).dy);

			valuesInv.push_back(1.0 / (CellsV(i, j).dx * CellsV(i, j).dy));

			++eqCounter;
		}
	}

	arma::Mat<arma::uword> M1(rowIndices);
	arma::Mat<arma::uword> M2(columnIndices);
	arma::Col<double> vals(values);
	arma::Col<double> valsInv(valuesInv);

	return std::pair<arma::SpMat<double>, arma::SpMat<double>>(
		{
			arma::SpMat<double>(arma::join_cols(M1.t(), M2.t()), vals, m_mesh.getNumU() + m_mesh.getNumV(), m_mesh.getNumU() + m_mesh.getNumV()),
			arma::SpMat<double>(arma::join_cols(M1.t(), M2.t()), valsInv, m_mesh.getNumU() + m_mesh.getNumV(), m_mesh.getNumU() + m_mesh.getNumV())
		}
	);

}