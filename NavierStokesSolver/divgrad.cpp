#ifndef H_DIVGRAD
#define H_DIVGRAD

#include <iostream>
#include <armadillo>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mesh.h"
#include "solver.h"
#include "boundary.h"

arma::SpMat<double> solver::setupDivergenceMatrix() {

	const arma::field<cell>& CellsU = m_mesh.getCellsU();
	const arma::field<cell>& CellsV = m_mesh.getCellsV();
	const arma::field<cell>& CellsP = m_mesh.getCellsP();

	std::vector<arma::uword> rowIndices;
	std::vector<arma::uword> columnIndices;
	std::vector<double> values;

	std::vector<int> di = { 1, 0, -1, 0 };
	std::vector<int> dj = { 0, 1, 0, -1 };

	arma::uword eqCounter = 0;

	bool interiorNeighbour_flag = true;

	//for all pressure unknowns
	for (arma::uword i = 0; i < m_mesh.getNumCellsY(); ++i) {
		for (arma::uword j = 0; j < m_mesh.getNumCellsX(); ++j) {

			//if cell is not on the boundary
			if (!CellsP(i, j).onBoundary) {

				//for all velocity components
				for (int k = 0; k < 4; ++k) {

					rowIndices.push_back(eqCounter);
					
					//if y direction
					if (k == 0 || k == 2) {
						columnIndices.push_back(CellsV(i + roundl(0.5 * di[k] + 0.5), j).vectorIndex);
						values.push_back(di[k] * CellsP(i, j).dx);
					}
						
					//if x direction
					if (k == 1 || k == 3) {
						columnIndices.push_back(CellsU(i, j + roundl(0.5 * dj[k] + 0.5)).vectorIndex);
						values.push_back(dj[k] * CellsP(i, j).dy);
					}

				}

			}
			//if on boundary
			else {

				//for all velocity components
				for (int k = 0; k < 4; ++k) {

					//deal with y components
					if (k == 0 || k == 2) {

						auto it = std::find_if(CellsV(i + roundl(0.5 * di[k] + 0.5), j).boundaryData.begin(), CellsV(i + roundl(0.5 * di[k] + 0.5), j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							});

						//if component on upper or lower boundary
						if (it != CellsV(i + roundl(0.5 * di[k] + 0.5), j).boundaryData.end()) {

							//switch over possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_UL):

								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsV(0, j).vectorIndex);
								values.push_back(di[k] * CellsP(i, j).dx);

								break;
							}

							interiorNeighbour_flag = false;

						}
						else {

							interiorNeighbour_flag = true;

						}

					}

					//deal with x components
					if (k == 1 || k == 3) {

						auto it = std::find_if(CellsU(i, j + roundl(0.5 * dj[k] + 0.5)).boundaryData.begin(), CellsU(i, j + roundl(0.5 * dj[k] + 0.5)).boundaryData.end(), [k](_boundaryData data) {
							return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
							});

						//if component on left or right boundary
						if (it != CellsU(i, j + roundl(0.5 * dj[k] + 0.5)).boundaryData.end()) {

							//switch over possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_LR):

								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsU(i, 0).vectorIndex);
								values.push_back(dj[k] * CellsP(i, j).dy);

								break;
							}

							interiorNeighbour_flag = false;

						}
						else {

							interiorNeighbour_flag = true;

						}

					}

					if (interiorNeighbour_flag) {
						rowIndices.push_back(eqCounter);

						//if y direction
						if (k == 0 || k == 2) {
							columnIndices.push_back(CellsV(i + roundl(0.5 * di[k] + 0.5), j).vectorIndex);
							values.push_back(di[k] * CellsP(i, j).dx);
						}

						//if x direction
						if (k == 1 || k == 3) {
							columnIndices.push_back(CellsU(i, j + roundl(0.5 * dj[k] + 0.5)).vectorIndex);
							values.push_back(dj[k] * CellsP(i, j).dy);
						}
					}

				}

			}

			++eqCounter;
		}
	}

	arma::Mat<arma::uword> M1(rowIndices);
	arma::Mat<arma::uword> M2(columnIndices);
	arma::Col<double> vals(values);

	return arma::SpMat<double>(arma::join_cols(M1.t(), M2.t()), vals, m_mesh.getNumCellsX() * m_mesh.getNumCellsY(), m_mesh.getNumU() + m_mesh.getNumV());
}


#endif