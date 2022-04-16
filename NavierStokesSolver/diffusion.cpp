#include <iostream>
#include <armadillo>
#include <vector>
#include <algorithm>

#include "mesh.h"
#include "solver.h"
#include "boundary.h"

arma::SpMat<double> solver::setupDiffusionMatrix() {

	const arma::field<cell>& CellsU = m_mesh.getCellsU();
	const arma::field<cell>& CellsV = m_mesh.getCellsV();

	std::vector<arma::uword> rowIndices;
	std::vector<arma::uword> columnIndices;
	std::vector<double> values;

	std::vector<int> di = { 1, 0, -1, 0 };
	std::vector<int> dj = { 0, 1, 0, -1 };

	double matrixCentralComponent = 0.0;
	double component;

	arma::uword eqCounter = 0;

	bool interiorNeighbour_flag = true;

	//for all cells containing a U unknown
	for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
		for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

			//if interior node
			if (!CellsU(i, j).onBoundary) {

				//loop over neighbours involved in computational stencil
				for (int k = 0; k < 4; ++k) {
					
					//if neighbour not on boundary or has wall parallel to flow component
					if (!CellsU(i + di[k], j + dj[k]).onBoundary || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'N' || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'S') {
						
						rowIndices.push_back(eqCounter);
						columnIndices.push_back(CellsU(i + di[k], j + dj[k]).vectorIndex);

						component = (di[k] != 0) ? di[k] * CellsU(i, j).dx / (CellsU(i + di[k], j + dj[k]).y - CellsU(i, j).y) : dj[k] * CellsU(i, j).dy / (CellsU(i + di[k], j + dj[k]).x - CellsU(i, j).x);
						matrixCentralComponent -= component;

						values.push_back(component);

					}
					//if neighbour on boundary normal to flow component 
					else {

						//switch over all possible boundary conditions
						switch (CellsU(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::NO_SLIP):
							component = (di[k] != 0) ? di[k] * CellsU(i, j).dx / (CellsU(i + di[k], j + dj[k]).y - CellsU(i, j).y) : dj[k] * CellsU(i, j).dy / (CellsU(i + di[k], j + dj[k]).x - CellsU(i, j).x);
							matrixCentralComponent -= component;
							break;
						case(B_CONDITION::PERIODIC_LR):
							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, 0).vectorIndex);

							component = (di[k] != 0) ? di[k] * CellsU(i, j).dx / (CellsU(i + di[k], j + dj[k]).y - CellsU(i, j).y) : dj[k] * CellsU(i, j).dy / (CellsU(i + di[k], j + dj[k]).x - CellsU(i, j).x);
							matrixCentralComponent -= component;

							values.push_back(component);
							break;
						}

					}
				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;
			}
			//if boundary node
			else {

				for (int k = 0; k < 4; ++k) {

					// deal with vertical boundaries
					if (k == 0 || k == 2) {

						auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							}
						);

						//if it has boundary south or north
						if (it != CellsU(i, j).boundaryData.end()) {

							//switch over possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::NO_SLIP):
								component = CellsU(i, j).dx / (0.5 * CellsU(i, j).dy);
								matrixCentralComponent -= component;
								break;
							case(B_CONDITION::PERIODIC_UL):
								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsU(((it->boundaryDir == 'N') ? 0 : (m_mesh.getEndIndUy() - 1)), j).vectorIndex);

								component = CellsU(i, j).dx / (0.5 * CellsU(0, j).dy + 0.5 * CellsU(m_mesh.getEndIndUy() - 1, j).dy);
								matrixCentralComponent -= component;

								values.push_back(component);
								break;
							}

							interiorNeighbour_flag = false;
						}
						else {
							interiorNeighbour_flag = true;
						}
					}

					//deal with left and right boundaries
					if (k == 1 || k == 3) {

						auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
							}
						);

						//if it has boundary left or right
						if (it != CellsU(i, j).boundaryData.end()) {

							//switch over possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_LR):
								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsU(i, m_mesh.getEndIndUx() - 1).vectorIndex);

								component = CellsU(i, j).dy / (m_mesh.getLengthX() - CellsU(i, m_mesh.getEndIndUx() - 1).x);
								matrixCentralComponent -= component;

								values.push_back(component);
								break;
							}

							interiorNeighbour_flag = false;
						}
						else {
							interiorNeighbour_flag = true;
						}


						//deal with corner cases on top and bottom boundary (getNumCellsX-1 gives second to last node)
						if (j == 1 || j == (m_mesh.getNumCellsX() - 1)) {

							auto it = std::find_if(CellsU(i, j + dj[k]).boundaryData.begin(), CellsU(i, j + dj[k]).boundaryData.end(), [k](_boundaryData data) {
									return (data.boundaryDir == 'E') || (data.boundaryDir == 'W');
								});

							//if left or right neighbour on left or right boundary
							if (it != CellsU(i, j + dj[k]).boundaryData.end()) {
								switch (it->B_TYPE) {
								case(B_CONDITION::NO_SLIP):
									component = (di[k] != 0) ? di[k] * CellsU(i, j).dx / (CellsU(i + di[k], j + dj[k]).y - CellsU(i, j).y) : dj[k] * CellsU(i, j).dy / (CellsU(i + di[k], j + dj[k]).x - CellsU(i, j).x);
									matrixCentralComponent -= component;
									break;
								case(B_CONDITION::PERIODIC_LR):
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i, 0).vectorIndex);

									component = (di[k] != 0) ? di[k] * CellsU(i, j).dx / (CellsU(i + di[k], j + dj[k]).y - CellsU(i, j).y) : dj[k] * CellsU(i, j).dy / (CellsU(i + di[k], j + dj[k]).x - CellsU(i, j).x);
									matrixCentralComponent -= component;

									values.push_back(component);
									break;
								}

								interiorNeighbour_flag = false;
							}
							else {
								interiorNeighbour_flag = true;
							}
						}
					}

					//deal with neighbours that have interior discretization (note: may still be on boundary, just no effect on discretization)
					if (interiorNeighbour_flag) {
						rowIndices.push_back(eqCounter);
						columnIndices.push_back(CellsU(i + di[k], j + dj[k]).vectorIndex);

						component = (di[k] != 0) ? di[k] * CellsU(i, j).dx / (CellsU(i + di[k], j + dj[k]).y - CellsU(i, j).y) : dj[k] * CellsU(i, j).dy / (CellsU(i + di[k], j + dj[k]).x - CellsU(i, j).x);
						matrixCentralComponent -= component;

						values.push_back(component);
					}

				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;

			}

			++eqCounter;
		}
	}

	//for all cells containing a V unknown
	for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
		for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

			//if interior node
			if (!CellsV(i, j).onBoundary) {

				//loop over neighbours involved in computational stencil
				for (int k = 0; k < 4; ++k) {

					//if neighbour not on boundary or has wall parallel to flow component
					if (!CellsV(i + di[k], j + dj[k]).onBoundary || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'E' || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'W') {

						rowIndices.push_back(eqCounter);
						columnIndices.push_back(CellsV(i + di[k], j + dj[k]).vectorIndex);

						component = (di[k] != 0) ? di[k] * CellsV(i, j).dx / (CellsV(i + di[k], j + dj[k]).y - CellsV(i, j).y) : dj[k] * CellsV(i, j).dy / (CellsV(i + di[k], j + dj[k]).x - CellsV(i, j).x);
						matrixCentralComponent -= component;

						values.push_back(component);

					}
					//if neighbour on boundary normal to flow component 
					else {

						//switch over all possible boundary conditions
						switch (CellsV(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::NO_SLIP):
							component = (di[k] != 0) ? di[k] * CellsV(i, j).dx / (CellsV(i + di[k], j + dj[k]).y - CellsV(i, j).y) : dj[k] * CellsV(i, j).dy / (CellsV(i + di[k], j + dj[k]).x - CellsV(i, j).x);
							matrixCentralComponent -= component;
							break;
						case(B_CONDITION::PERIODIC_UL):
							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(0, j).vectorIndex);

							component = (di[k] != 0) ? di[k] * CellsV(i, j).dx / (CellsV(i + di[k], j + dj[k]).y - CellsV(i, j).y) : dj[k] * CellsV(i, j).dy / (CellsV(i + di[k], j + dj[k]).x - CellsV(i, j).x);
							matrixCentralComponent -= component;

							values.push_back(component);
							break;
						}

					}
				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;
			}
			//if boundary cell
			else {

				//loop over neighbours
				for (int k = 0; k < 4; ++k) {

					//deal with vertical boundaries
					if (k == 0 || k == 2) {

						auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							});

						//if it on boundary south or north
						if (it != CellsV(i, j).boundaryData.end()) {

							//switch over possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_UL):
								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsV(m_mesh.getEndIndVy() - 1, j).vectorIndex);

								component = CellsV(i, j).dx / (m_mesh.getLengthY() - CellsV(m_mesh.getEndIndVy() - 1, j).y);
								matrixCentralComponent -= component;

								values.push_back(component);
								break;
							}

							interiorNeighbour_flag = false;
						}
						else {
							interiorNeighbour_flag = true;
						}


						//deal with corner cases on left and right boundary
						if (i == 1 || i == (m_mesh.getNumCellsY() - 1)) {

							auto it = std::find_if(CellsV(i + di[k], j).boundaryData.begin(), CellsV(i + di[k], j).boundaryData.end(), [k](_boundaryData data) {
									return (data.boundaryDir == 'N') || (data.boundaryDir == 'S');
								});

							//if top or bottom neighbour on top or bottom boundary
							if (it != CellsV(i + di[k], j).boundaryData.end()) {
								switch (it->B_TYPE) {
								case(B_CONDITION::NO_SLIP):
									component = (di[k] != 0) ? di[k] * CellsV(i, j).dx / (CellsV(i + di[k], j + dj[k]).y - CellsV(i, j).y) : dj[k] * CellsV(i, j).dy / (CellsV(i + di[k], j + dj[k]).x - CellsV(i, j).x);
									matrixCentralComponent -= component;
									break;
								case(B_CONDITION::PERIODIC_UL):
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j).vectorIndex);

									component = (di[k] != 0) ? di[k] * CellsV(i, j).dx / (CellsV(i + di[k], j + dj[k]).y - CellsV(i, j).y) : dj[k] * CellsV(i, j).dy / (CellsV(i + di[k], j + dj[k]).x - CellsV(i, j).x);
									matrixCentralComponent -= component;

									values.push_back(component);
									break;
								}

								interiorNeighbour_flag = false;
							}
							else {
								interiorNeighbour_flag = true;
							}
						}

					}

					//deal with left and right boundary
					if (k == 1 || k == 3) {

						auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
							});

						//if cell on boundary left or right
						if (it != CellsV(i, j).boundaryData.end()) {

							//switch over possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::NO_SLIP):
								component = CellsV(i, j).dy / (0.5 * CellsV(i, j).dx);
								matrixCentralComponent -= component;
								break;
							case(B_CONDITION::PERIODIC_LR):
								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsV(i, ((it->boundaryDir == 'E') ? 0 : (m_mesh.getEndIndVx() - 1))).vectorIndex);

								component = CellsV(i, j).dy / (0.5 * CellsV(i, 0).dx + 0.5 * CellsV(i, m_mesh.getEndIndVx() - 1).dx);
								matrixCentralComponent -= component;

								values.push_back(component);
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
						columnIndices.push_back(CellsV(i + di[k], j + dj[k]).vectorIndex);

						component = (di[k] != 0) ? di[k] * CellsV(i, j).dx / (CellsV(i + di[k], j + dj[k]).y - CellsV(i, j).y) : dj[k] * CellsV(i, j).dy / (CellsV(i + di[k], j + dj[k]).x - CellsV(i, j).x);
						matrixCentralComponent -= component;

						values.push_back(component);
					}

				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;
			}

			++eqCounter;
		}
	}


	arma::Mat<arma::uword> M1(rowIndices);
	arma::Mat<arma::uword> M2(columnIndices);
	arma::Col<double> vals(values);

	return arma::SpMat<double>(arma::join_cols(M1.t(), M2.t()), vals, m_mesh.getNumU() + m_mesh.getNumV(), m_mesh.getNumU() + m_mesh.getNumV());
}

