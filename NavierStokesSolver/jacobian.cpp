#include <iostream>
#include <armadillo>
#include <vector>
#include <algorithm>

#include "mesh.h"
#include "solver.h"
#include "boundary.h"

#include "ROM.h"


arma::SpMat<double> solver::J(const arma::Col<double>& vel) const {

	const arma::field<cell>& CellsU = m_mesh.getCellsU();
	const arma::field<cell>& CellsV = m_mesh.getCellsV();

	std::vector<int> di = { 1, 0, -1, 0 };
	std::vector<int> dj = { 0, 1, 0, -1 };

	std::vector<int> diu = { 1, 1, 0, 0 };
	std::vector<int> dju = { -1, 0, 0, -1 };

	std::vector<int> div = {0, 0, -1, -1};
	std::vector<int> djv = { 0, 1, 1, 0 };

	bool interiorNeighbour_flag = true;

	std::vector<arma::uword> rowIndices;
	std::vector<arma::uword> columnIndices;
	std::vector<double> values;

	double matrixCentralComponent = 0.0;
	double component;

	arma::uword eqCounter = 0;

	int index = 0;

	//for all cells containing a U unknown
	for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
		for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

			//if interior node
			if (!CellsU(i, j).onBoundary) {

				//loop over neighbours involved in computational stencil
				for (int k = 0; k < 4; ++k) {

					//if neighbour not on boundary or has wall parallel to flow component
					if (!CellsU(i + di[k], j + dj[k]).onBoundary || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'N' || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'S') {

						switch (di[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i + di[k], j).vectorIndex);

							index = (di[k] == 1) ? 1 : 0;

							values.push_back(di[k] * 0.25 * (vel(CellsV(i + index, j - 1).vectorIndex) + vel(CellsV(i + index, j).vectorIndex)) * CellsU(i, j).dx);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, j + dj[k]).vectorIndex);

							values.push_back(dj[k] * (0.25 * vel(CellsU(i,j).vectorIndex) + 0.5 * vel(CellsU(i, j + dj[k]).vectorIndex)) * CellsU(i, j).dy);

							matrixCentralComponent += dj[k] * 0.25 * vel(CellsU(i, j + dj[k]).vectorIndex) * CellsU(i, j).dy;

							break;

						}
							

					}
					//if neighbour on boundary
					else {

						//switch over all possible boundary conditions
						switch (CellsU(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::PERIODIC_LR):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, 0).vectorIndex);

							values.push_back(dj[k] * (0.25 * vel(CellsU(i, j).vectorIndex) + 0.5 * vel(CellsU(i, 0).vectorIndex)) * CellsU(i, j).dy);

							matrixCentralComponent += dj[k] * 0.25 * vel(CellsU(i, 0).vectorIndex) * CellsU(i, j).dy;

							break;
						}

					}

				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;

				//jacobian components of y velocities
				for (int k = 0; k < 4; ++k) {

					rowIndices.push_back(eqCounter);
					columnIndices.push_back(CellsV(i + diu[k], j + dju[k]).vectorIndex);

					switch (diu[k] == 1) {
					case (true):

						values.push_back(0.25 * vel(CellsU(i + 1, j).vectorIndex) * CellsU(i, j).dx);

						break;
					case(false):
						
						values.push_back(-0.25 * vel(CellsU(i - 1, j).vectorIndex) * CellsU(i, j).dx);

						break;
					}

				}

			}
			//if boundary node
			else {

				//loop over neighbours
				for (int k = 0; k < 4; ++k) {

					//deal with upward and downward stuff
					if (k == 0 || k == 2) {

						//deal with left and right boundaries separately
						if (j != 0 && j != m_mesh.getNumCellsX()) {

							auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
								});

							//if stuff on upper or lower boundar
							if (it != CellsU(i, j).boundaryData.end()) {

								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									//uN or uS
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU((di[k] == 1) ? 0 : (m_mesh.getNumCellsY() - 1), j).vectorIndex);

									values.push_back(di[k] * 0.25 * (vel(CellsV(0, j - 1).vectorIndex) + vel(CellsV(0, j).vectorIndex)) * CellsU(i, j).dx);

									//vNW or vSW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j - 1).vectorIndex);

									values.push_back(di[k] * 0.25 * vel(CellsU((di[k] == 1) ? 0 : (m_mesh.getNumCellsY() - 1), j).vectorIndex) * CellsU(i, j).dx);

									//vNE or VSE
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j).vectorIndex);

									values.push_back(di[k] * 0.25 * vel(CellsU((di[k] == 1) ? 0 : (m_mesh.getNumCellsY() - 1), j).vectorIndex) * CellsU(i, j).dx);

									break;
								}

								interiorNeighbour_flag = false;
							}
							else {

								//discretize 3 upper or lower unknowns as interior unknowns
								interiorNeighbour_flag = true;
							
							}

						}
						//deal with left and right boundary
						else {

							auto it1 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
								});

							auto it2 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [](_boundaryData data) {
								return data.boundaryDir == 'E' || data.boundaryDir == 'W';
								});

							//if on upper or lower boundary
							if (it1 != CellsU(i, j).boundaryData.end()) {

								//switch over left and right boundary conditions
								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									//switch over upper and lower boundary conditions
									switch (it1->B_TYPE) {
									case(B_CONDITION::PERIODIC_UL):

										//uN or uS
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsU((di[k] == 1) ? 0 : (m_mesh.getNumCellsY() - 1), j).vectorIndex);

										values.push_back(di[k] * 0.25 * (vel(CellsV(0, 0).vectorIndex) + vel(CellsV(0, (m_mesh.getNumCellsX() - 1)).vectorIndex)) * CellsU(i, j).dx);

										//vNW or vSW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsV(0, (m_mesh.getNumCellsX() - 1)).vectorIndex);

										values.push_back(di[k] * 0.25 * vel(CellsU((di[k] == 1) ? 0 : (m_mesh.getNumCellsY() - 1), j).vectorIndex) * CellsU(i, j).dx);

										//vNE or VSE
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsV(0, 0).vectorIndex);

										values.push_back(di[k] * 0.25 * vel(CellsU((di[k] == 1) ? 0 : (m_mesh.getNumCellsY() - 1), j).vectorIndex) * CellsU(i, j).dx);

										break;
									}

									break;
								}

								
							}
							//if not on upper or lower boundary
							else {

								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									//uN or uS
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i + di[k], j).vectorIndex);

									values.push_back(di[k] * 0.25 * (vel(CellsV(i + ((di[k] == 1) ? 1 : 0) , 0).vectorIndex) + vel(CellsV(i + ((di[k] == 1) ? 1 : 0), m_mesh.getNumCellsX() - 1).vectorIndex))* CellsU(i, j).dx);

									//vNW or vSW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), 0).vectorIndex);

									values.push_back(di[k] * 0.25 * vel(CellsU(i + di[k], 0).vectorIndex) * CellsU(i, j).dx);

									//vNE or VSE
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), (m_mesh.getNumCellsX() - 1)).vectorIndex);

									values.push_back(di[k] * 0.25 * vel(CellsU(i + di[k], 0).vectorIndex) * CellsU(i, j).dx);

									break;
								}

							}

							interiorNeighbour_flag = false;

						}

					}

					//deal with left, right and central stuff
					if (k == 1 || k == 3) {

						auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
							});

						//if on left or right boundary
						if (it != CellsU(i, j).boundaryData.end()) {

							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_LR):

								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsU(i, (m_mesh.getNumCellsX() - 1)).vectorIndex);

								values.push_back(-(0.5 * vel(CellsU(i, (m_mesh.getNumCellsX() - 1)).vectorIndex) + 0.25 * vel(CellsU(i, j).vectorIndex)) * CellsU(i, j).dy);

								matrixCentralComponent += -0.25 * vel(CellsU(i, (m_mesh.getNumCellsX() - 1)).vectorIndex) * CellsU(i, j).dy;

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

								//switch over all possible boundary conditions
								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i, 0).vectorIndex);

									values.push_back(dj[k] * (0.25 * vel(CellsU(i, j).vectorIndex) + 0.5 * vel(CellsU(i, 0).vectorIndex))* CellsU(i, j).dy);

									matrixCentralComponent += dj[k] * 0.25 * vel(CellsU(i, 0).vectorIndex) * CellsU(i, j).dy;

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}
						}

					}

					
					if (interiorNeighbour_flag) {

						switch (di[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i + di[k], j).vectorIndex);

							index = (di[k] == 1) ? 1 : 0;

							values.push_back(di[k] * 0.25 * (vel(CellsV(i + index, j - 1).vectorIndex) + vel(CellsV(i + index, j).vectorIndex)) * CellsU(i, j).dx);

							//vNW or vSW
							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), j - 1).vectorIndex);

							values.push_back(di[k] * 0.25 * vel(CellsU(i + di[k], j).vectorIndex) * CellsU(i, j).dx);

							//vNE or VSE
							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex);

							values.push_back(di[k] * 0.25 * vel(CellsU(i + di[k], j).vectorIndex) * CellsU(i, j).dx);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, j + dj[k]).vectorIndex);

							values.push_back(dj[k] * (0.25 * vel(CellsU(i, j).vectorIndex) + 0.5 * vel(CellsU(i, j + dj[k]).vectorIndex)) * CellsU(i, j).dy);

							matrixCentralComponent += dj[k] * 0.25 * vel(CellsU(i, j + dj[k]).vectorIndex) * CellsU(i, j).dy;

							break;

						}

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

	//loop over all V unknowns
	for (arma::uword i = m_mesh.getStartIndVy(); i < m_mesh.getEndIndVy(); ++i) {
		for (arma::uword j = m_mesh.getStartIndVx(); j < m_mesh.getEndIndVx(); ++j) {

			//if interior node
			if (!CellsV(i, j).onBoundary) {

				//loop over neigbouring cells
				for (int k = 0; k < 4; ++k) {

					//if neighbour not on upper or lower boundary calculate neighbour's contribution to convection operator
					if (!CellsV(i + di[k], j + dj[k]).onBoundary || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'E' || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'W') {
					
						switch (dj[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i, j + dj[k]).vectorIndex);

							index = (dj[k] == 1) ? 1 : 0;

							values.push_back(dj[k] * 0.25 * (vel(CellsU(i, j + index).vectorIndex) + vel(CellsU(i-1, j + index).vectorIndex)) * CellsV(i, j).dy);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + di[k], j).vectorIndex);

							values.push_back(di[k] * (0.5 * vel(CellsV(i + di[k], j).vectorIndex) + 0.25 * vel(CellsV(i, j).vectorIndex))* CellsV(i, j).dx);

							matrixCentralComponent += di[k] * 0.25 * vel(CellsV(i + di[k], j).vectorIndex) * CellsV(i, j).dx;

							break;

						}

					}
					//if neighbour on boundary
					else {

						//switch over all possible boundary conditions
						switch (CellsV(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::PERIODIC_UL):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(0, j).vectorIndex);

							values.push_back(di[k] * (0.25 * vel(CellsV(i, j).vectorIndex) + 0.5 * vel(CellsV(0, j).vectorIndex)) * CellsV(i, j).dx);

							matrixCentralComponent += di[k] * 0.25 * vel(CellsV(0, j).vectorIndex) * CellsV(i, j).dx;

							break;
						}
					}
				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;

				//jacobian components of u velocities
				for (int k = 0; k < 4; ++k) {

					rowIndices.push_back(eqCounter);
					columnIndices.push_back(CellsU(i + div[k], j + djv[k]).vectorIndex);

					switch (djv[k] == 1) {
					case(true):

						values.push_back(0.25 * vel(CellsV(i, j + 1).vectorIndex) * CellsV(i, j).dy);

						break;
					case(false):

						values.push_back(-0.25 * vel(CellsV(i, j - 1).vectorIndex) * CellsV(i, j).dy);

						break;
					}


				}

			}
			//if on boundary
			else {

				//loop over neighbours
				for (int k = 0; k < 4; ++k) {

					//deal with upper and lower stuff
					if (k == 0 || k == 2) {

						auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							});

						//if on upper or lower boundary
						if (it != CellsV(i, j).boundaryData.end()) {

							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_UL):

								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsV(m_mesh.getNumCellsY() - 1, j).vectorIndex);

								values.push_back(-(0.5 * vel(CellsV(m_mesh.getNumCellsY() - 1, j).vectorIndex) + 0.25 * vel(CellsV(i, j).vectorIndex)) * CellsV(i, j).dx);

								matrixCentralComponent += -0.25 * vel(CellsV(m_mesh.getNumCellsY() - 1, j).vectorIndex) * CellsV(i, j).dx;

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

							//if top or bottom neighbour on top or bottom boundary resp.
							if (it != CellsV(i + di[k], j).boundaryData.end()) {

								//switch over all possible boundary conditions
								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j).vectorIndex);

									values.push_back(di[k] * (0.25 * vel(CellsV(i, j).vectorIndex) + 0.5 * vel(CellsV(0, j).vectorIndex))* CellsV(i, j).dx);

									matrixCentralComponent += di[k] * 0.25 * vel(CellsV(0, j).vectorIndex) * CellsV(i, j).dx;

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}

						}

					}

					//deal with left and right stuff
					if (k == 1 || k == 3) {

						//deal with upper and lower boundaries in separate code
						if (i != 0 && i != m_mesh.getNumCellsY()) {

							auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
								});

							//if flux on left or right boundary
							if (it != CellsV(i, j).boundaryData.end()) {

								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									//vE or vW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(i, (dj[k] == 1) ? 0 : (m_mesh.getNumCellsX() - 1)).vectorIndex);

									values.push_back(dj[k] * 0.25 * (vel(CellsU(i, 0).vectorIndex) + vel(CellsU(i - 1, 0).vectorIndex)) * CellsV(i, j).dy);

									//uNE or uNW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i, 0).vectorIndex);

									values.push_back(dj[k] * 0.25 * vel(CellsV(i, (dj[k] == 1) ? 0 : (m_mesh.getNumCellsX() - 1)).vectorIndex) * CellsV(i, j).dy);

									//uSE or uSW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i - 1, 0).vectorIndex);

									values.push_back(dj[k] * 0.25 * vel(CellsV(i, (dj[k] == 1) ? 0 : (m_mesh.getNumCellsX() - 1)).vectorIndex) * CellsV(i, j).dy);

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}

						}
						//deal with upper and lower boundary
						else {

							auto it1 = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
								});

							auto it2 = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [](_boundaryData data) {
								return data.boundaryDir == 'N' || data.boundaryDir == 'S';
								});

							//if flux on left or right boundary
							if (it1 != CellsV(i, j).boundaryData.end()) {

								//switch over upper and lower boundary conditions
								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									//switch over right and left boundary conditions
									switch (it1->B_TYPE) {
									case(B_CONDITION::PERIODIC_LR):

										//vE or vW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsV(0, (dj[k] == 1) ? 0 : (m_mesh.getNumCellsX() - 1)).vectorIndex);

										values.push_back(dj[k] * 0.25 * (vel(CellsU(0, 0).vectorIndex) + vel(CellsU(m_mesh.getNumCellsY() - 1, 0).vectorIndex)) * CellsV(i, j).dy);

										//uNE or uNW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsU(0, 0).vectorIndex);

										values.push_back(dj[k] * 0.25 * vel(CellsV(0, (dj[k] == 1) ? 0 : (m_mesh.getNumCellsX() - 1)).vectorIndex) * CellsV(i, j).dy);
									
										//uSE or uSW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsU(m_mesh.getNumCellsY() - 1, 0).vectorIndex);

										values.push_back(dj[k] * 0.25 * vel(CellsV(0, (dj[k] == 1) ? 0 : (m_mesh.getNumCellsX() - 1)).vectorIndex) * CellsV(i, j).dy);
										break;
									}

									break;
								}

							}
							//if flux not on left or right boundary
							else {

								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j + dj[k]).vectorIndex);

									values.push_back(dj[k] * 0.25 * (vel(CellsU(0, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) + vel(CellsU(m_mesh.getNumCellsY() - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex)) * CellsV(i, j).dy);

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(0, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex);

									values.push_back(dj[k] * 0.25 * vel(CellsV(0, j + dj[k]).vectorIndex) * CellsV(i, j).dy);

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(m_mesh.getNumCellsY() - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex);

									values.push_back(dj[k] * 0.25 * vel(CellsV(0, j + dj[k]).vectorIndex) * CellsV(i, j).dy);

									break;
								}

							}

							interiorNeighbour_flag = false;

						}

					}

					if (interiorNeighbour_flag) {

						switch (dj[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i, j + dj[k]).vectorIndex);

							index = (dj[k] == 1) ? 1 : 0;

							values.push_back(dj[k] * 0.25 * (vel(CellsU(i, j + index).vectorIndex) + vel(CellsU(i - 1, j + index).vectorIndex)) * CellsV(i, j).dy);

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, j + index).vectorIndex);

							values.push_back(dj[k] * 0.25 * vel(CellsV(i, j + dj[k]).vectorIndex) * CellsV(i, j).dy);

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i - 1, j + index).vectorIndex);

							values.push_back(dj[k] * 0.25 * vel(CellsV(i, j + dj[k]).vectorIndex) * CellsV(i, j).dy);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + di[k], j).vectorIndex);

							values.push_back(di[k] * (0.5 * vel(CellsV(i + di[k], j).vectorIndex) + 0.25 * vel(CellsV(i, j).vectorIndex)) * CellsV(i, j).dx);

							matrixCentralComponent += di[k] * 0.25 * vel(CellsV(i + di[k], j).vectorIndex) * CellsV(i, j).dx;

							break;

						}
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

	//can this really not be done with std::vectors???
	arma::Mat<arma::uword> M1(rowIndices);
	arma::Mat<arma::uword> M2(columnIndices);
	arma::Col<double> vals(values);

	return arma::SpMat<double>(arma::join_cols(M1.t(), M2.t()), vals, m_mesh.getNumU() + m_mesh.getNumV(), m_mesh.getNumU() + m_mesh.getNumV());
}



arma::Row<double> ROM_Solver::Jindex(const arma::Col<double>& a, arma::uword vecIndex, arma::uword i, arma::uword j) const {

	const arma::field<cell>& CellsU = m_solver.getMesh().getCellsU();
	const arma::field<cell>& CellsV = m_solver.getMesh().getCellsV();

	std::vector<int> di = { 1, 0, -1, 0 };
	std::vector<int> dj = { 0, 1, 0, -1 };

	std::vector<int> diu = { 1, 1, 0, 0 };
	std::vector<int> dju = { -1, 0, 0, -1 };

	std::vector<int> div = { 0, 0, -1, -1 };
	std::vector<int> djv = { 0, 1, 1, 0 };

	bool interiorNeighbour_flag = true;

	std::vector<arma::uword> rowIndices;
	std::vector<arma::uword> columnIndices;
	std::vector<double> values;

	double matrixCentralComponent = 0.0;
	double component;

	arma::uword eqCounter = 0;

	int index = 0;

	//arma::as_scalar(m_Psi.row(

	//if u node
	if (vecIndex < m_solver.getMesh().getNumU()) {

			//if interior node
			if (!CellsU(i, j).onBoundary) {

				//loop over neighbours involved in computational stencil
				for (int k = 0; k < 4; ++k) {

					//if neighbour not on boundary or has wall parallel to flow component
					if (!CellsU(i + di[k], j + dj[k]).onBoundary || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'N' || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'S') {

						switch (di[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i + di[k], j).vectorIndex);

							index = (di[k] == 1) ? 1 : 0;

							values.push_back(di[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsV(i + index, j - 1).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i + index, j).vectorIndex) * a)) * CellsU(i, j).dx);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, j + dj[k]).vectorIndex);

							values.push_back(dj[k] * (0.25 * arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a) + 0.5 * arma::as_scalar(m_Psi.row(CellsU(i, j + dj[k]).vectorIndex) * a)) * CellsU(i, j).dy);

							matrixCentralComponent += dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i, j + dj[k]).vectorIndex) * a) * CellsU(i, j).dy;

							break;

						}


					}
					//if neighbour on boundary
					else {

						//switch over all possible boundary conditions
						switch (CellsU(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::PERIODIC_LR):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, 0).vectorIndex);

							values.push_back(dj[k] * (0.25 * arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a) + 0.5 * arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a)) * CellsU(i, j).dy);

							matrixCentralComponent += dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a) * CellsU(i, j).dy;

							break;
						}

					}

				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;

				//jacobian components of y velocities
				for (int k = 0; k < 4; ++k) {

					rowIndices.push_back(eqCounter);
					columnIndices.push_back(CellsV(i + diu[k], j + dju[k]).vectorIndex);

					switch (diu[k] == 1) {
					case (true):

						values.push_back(0.25 * arma::as_scalar(m_Psi.row(CellsU(i + 1, j).vectorIndex) * a) * CellsU(i, j).dx);

						break;
					case(false):

						values.push_back(-0.25 * arma::as_scalar(m_Psi.row(CellsU(i - 1, j).vectorIndex) * a) * CellsU(i, j).dx);

						break;
					}

				}

			}
			//if boundary node
			else {

				//loop over neighbours
				for (int k = 0; k < 4; ++k) {

					//deal with upward and downward stuff
					if (k == 0 || k == 2) {

						//deal with left and right boundaries separately
						if (j != 0 && j != m_solver.getMesh().getNumCellsX()) {

							auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
								});

							//if stuff on upper or lower boundar
							if (it != CellsU(i, j).boundaryData.end()) {

								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									//uN or uS
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU((di[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsY() - 1), j).vectorIndex);

									values.push_back(di[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsV(0, j - 1).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a)) * CellsU(i, j).dx);

									//vNW or vSW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j - 1).vectorIndex);

									values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU((di[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsY() - 1), j).vectorIndex) * a) * CellsU(i, j).dx);

									//vNE or VSE
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j).vectorIndex);

									values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU((di[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsY() - 1), j).vectorIndex) * a) * CellsU(i, j).dx);

									break;
								}

								interiorNeighbour_flag = false;
							}
							else {

								//discretize 3 upper or lower unknowns as interior unknowns
								interiorNeighbour_flag = true;

							}

						}
						//deal with left and right boundary
						else {

							auto it1 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
								});

							auto it2 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [](_boundaryData data) {
								return data.boundaryDir == 'E' || data.boundaryDir == 'W';
								});

							//if on upper or lower boundary
							if (it1 != CellsU(i, j).boundaryData.end()) {

								//switch over left and right boundary conditions
								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									//switch over upper and lower boundary conditions
									switch (it1->B_TYPE) {
									case(B_CONDITION::PERIODIC_UL):

										//uN or uS
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsU((di[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsY() - 1), j).vectorIndex);

										values.push_back(di[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsV(0, 0).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(0, (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex) * a)) * CellsU(i, j).dx);

										//vNW or vSW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsV(0, (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex);

										values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU((di[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsY() - 1), j).vectorIndex) * a) * CellsU(i, j).dx);

										//vNE or VSE
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsV(0, 0).vectorIndex);

										values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU((di[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsY() - 1), j).vectorIndex) * a) * CellsU(i, j).dx);

										break;
									}

									break;
								}


							}
							//if not on upper or lower boundary
							else {

								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									//uN or uS
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i + di[k], j).vectorIndex);

									values.push_back(di[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), 0).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), m_solver.getMesh().getNumCellsX() - 1).vectorIndex) * a)) * CellsU(i, j).dx);

									//vNW or vSW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), 0).vectorIndex);

									values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i + di[k], 0).vectorIndex) * a) * CellsU(i, j).dx);

									//vNE or VSE
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex);

									values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i + di[k], 0).vectorIndex) * a) * CellsU(i, j).dx);

									break;
								}

							}

							interiorNeighbour_flag = false;

						}

					}

					//deal with left, right and central stuff
					if (k == 1 || k == 3) {

						auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
							});

						//if on left or right boundary
						if (it != CellsU(i, j).boundaryData.end()) {

							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_LR):

								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsU(i, (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex);

								values.push_back(-(0.5 * arma::as_scalar(m_Psi.row(CellsU(i, (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex) * a) + 0.25 * arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a)) * CellsU(i, j).dy);

								matrixCentralComponent += -0.25 * arma::as_scalar(m_Psi.row(CellsU(i, (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex) * a) * CellsU(i, j).dy;

								break;
							}

							interiorNeighbour_flag = false;
						}
						else {

							interiorNeighbour_flag = true;

						}

						//deal with corner cases on top and bottom boundary (getNumCellsX-1 gives second to last node)
						if (j == 1 || j == (m_solver.getMesh().getNumCellsX() - 1)) {

							auto it = std::find_if(CellsU(i, j + dj[k]).boundaryData.begin(), CellsU(i, j + dj[k]).boundaryData.end(), [k](_boundaryData data) {
								return (data.boundaryDir == 'E') || (data.boundaryDir == 'W');
								});

							//if left or right neighbour on left or right boundary
							if (it != CellsU(i, j + dj[k]).boundaryData.end()) {

								//switch over all possible boundary conditions
								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i, 0).vectorIndex);

									values.push_back(dj[k] * (0.25 * arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a) + 0.5 * arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a)) * CellsU(i, j).dy);

									matrixCentralComponent += dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a) * CellsU(i, j).dy;

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}
						}

					}


					if (interiorNeighbour_flag) {

						switch (di[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i + di[k], j).vectorIndex);

							index = (di[k] == 1) ? 1 : 0;

							values.push_back(di[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsV(i + index, j - 1).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i + index, j).vectorIndex) * a)) * CellsU(i, j).dx);

							//vNW or vSW
							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), j - 1).vectorIndex);

							values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i + di[k], j).vectorIndex) * a) * CellsU(i, j).dx);

							//vNE or VSE
							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex);

							values.push_back(di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i + di[k], j).vectorIndex) * a) * CellsU(i, j).dx);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, j + dj[k]).vectorIndex);

							values.push_back(dj[k] * (0.25 * arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a) + 0.5 * arma::as_scalar(m_Psi.row(CellsU(i, j + dj[k]).vectorIndex) * a)) * CellsU(i, j).dy);

							matrixCentralComponent += dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsU(i, j + dj[k]).vectorIndex) * a) * CellsU(i, j).dy;

							break;

						}

					}

				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;
			}

			++eqCounter;
	}

	else{
			//if interior node
			if (!CellsV(i, j).onBoundary) {

				//loop over neigbouring cells
				for (int k = 0; k < 4; ++k) {

					//if neighbour not on upper or lower boundary calculate neighbour's contribution to convection operator
					if (!CellsV(i + di[k], j + dj[k]).onBoundary || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'E' || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'W') {

						switch (dj[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i, j + dj[k]).vectorIndex);

							index = (dj[k] == 1) ? 1 : 0;

							values.push_back(dj[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsU(i, j + index).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i - 1, j + index).vectorIndex) * a)) * CellsV(i, j).dy);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + di[k], j).vectorIndex);

							values.push_back(di[k] * (0.5 * arma::as_scalar(m_Psi.row(CellsV(i + di[k], j).vectorIndex) * a) + 0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a)) * CellsV(i, j).dx);

							matrixCentralComponent += di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(i + di[k], j).vectorIndex) * a) * CellsV(i, j).dx;

							break;

						}

					}
					//if neighbour on boundary
					else {

						//switch over all possible boundary conditions
						switch (CellsV(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::PERIODIC_UL):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(0, j).vectorIndex);

							values.push_back(di[k] * (0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a) + 0.5 * arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a)) * CellsV(i, j).dx);

							matrixCentralComponent += di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a) * CellsV(i, j).dx;

							break;
						}
					}
				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;

				//jacobian components of u velocities
				for (int k = 0; k < 4; ++k) {

					rowIndices.push_back(eqCounter);
					columnIndices.push_back(CellsU(i + div[k], j + djv[k]).vectorIndex);

					switch (djv[k] == 1) {
					case(true):

						values.push_back(0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j + 1).vectorIndex) * a) * CellsV(i, j).dy);

						break;
					case(false):

						values.push_back(-0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j - 1).vectorIndex) * a) * CellsV(i, j).dy);

						break;
					}


				}

			}
			//if on boundary
			else {

				//loop over neighbours
				for (int k = 0; k < 4; ++k) {

					//deal with upper and lower stuff
					if (k == 0 || k == 2) {

						auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							});

						//if on upper or lower boundary
						if (it != CellsV(i, j).boundaryData.end()) {

							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_UL):

								rowIndices.push_back(eqCounter);
								columnIndices.push_back(CellsV(m_solver.getMesh().getNumCellsY() - 1, j).vectorIndex);

								values.push_back(-(0.5 * arma::as_scalar(m_Psi.row(CellsV(m_solver.getMesh().getNumCellsY() - 1, j).vectorIndex) * a) + 0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a)) * CellsV(i, j).dx);

								matrixCentralComponent += -0.25 * arma::as_scalar(m_Psi.row(CellsV(m_solver.getMesh().getNumCellsY() - 1, j).vectorIndex) * a) * CellsV(i, j).dx;

								break;
							}

							interiorNeighbour_flag = false;

						}
						else {

							interiorNeighbour_flag = true;

						}

						//deal with corner cases on left and right boundary
						if (i == 1 || i == (m_solver.getMesh().getNumCellsY() - 1)) {

							auto it = std::find_if(CellsV(i + di[k], j).boundaryData.begin(), CellsV(i + di[k], j).boundaryData.end(), [k](_boundaryData data) {
								return (data.boundaryDir == 'N') || (data.boundaryDir == 'S');
								});

							//if top or bottom neighbour on top or bottom boundary resp.
							if (it != CellsV(i + di[k], j).boundaryData.end()) {

								//switch over all possible boundary conditions
								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j).vectorIndex);

									values.push_back(di[k] * (0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a) + 0.5 * arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a)) * CellsV(i, j).dx);

									matrixCentralComponent += di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a) * CellsV(i, j).dx;

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}

						}

					}

					//deal with left and right stuff
					if (k == 1 || k == 3) {

						//deal with upper and lower boundaries in separate code
						if (i != 0 && i != m_solver.getMesh().getNumCellsY()) {

							auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
								});

							//if flux on left or right boundary
							if (it != CellsV(i, j).boundaryData.end()) {

								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									//vE or vW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(i, (dj[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex);

									values.push_back(dj[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i - 1, 0).vectorIndex) * a)) * CellsV(i, j).dy);

									//uNE or uNW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i, 0).vectorIndex);

									values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(i, (dj[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex) * a) * CellsV(i, j).dy);

									//uSE or uSW
									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(i - 1, 0).vectorIndex);

									values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(i, (dj[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex) * a) * CellsV(i, j).dy);

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}

						}
						//deal with upper and lower boundary
						else {

							auto it1 = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
								});

							auto it2 = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [](_boundaryData data) {
								return data.boundaryDir == 'N' || data.boundaryDir == 'S';
								});

							//if flux on left or right boundary
							if (it1 != CellsV(i, j).boundaryData.end()) {

								//switch over upper and lower boundary conditions
								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									//switch over right and left boundary conditions
									switch (it1->B_TYPE) {
									case(B_CONDITION::PERIODIC_LR):

										//vE or vW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsV(0, (dj[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex);

										values.push_back(dj[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsU(0, 0).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(m_solver.getMesh().getNumCellsY() - 1, 0).vectorIndex) * a)) * CellsV(i, j).dy);

										//uNE or uNW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsU(0, 0).vectorIndex);

										values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(0, (dj[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex) * a) * CellsV(i, j).dy);

										//uSE or uSW
										rowIndices.push_back(eqCounter);
										columnIndices.push_back(CellsU(m_solver.getMesh().getNumCellsY() - 1, 0).vectorIndex);

										values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(0, (dj[k] == 1) ? 0 : (m_solver.getMesh().getNumCellsX() - 1)).vectorIndex) * a) * CellsV(i, j).dy);
										break;
									}

									break;
								}

							}
							//if flux not on left or right boundary
							else {

								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsV(0, j + dj[k]).vectorIndex);

									values.push_back(dj[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsU(0, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(m_solver.getMesh().getNumCellsY() - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a)) * CellsV(i, j).dy);

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(0, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex);

									values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(0, j + dj[k]).vectorIndex) * a) * CellsV(i, j).dy);

									rowIndices.push_back(eqCounter);
									columnIndices.push_back(CellsU(m_solver.getMesh().getNumCellsY() - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex);

									values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(0, j + dj[k]).vectorIndex) * a) * CellsV(i, j).dy);

									break;
								}

							}

							interiorNeighbour_flag = false;

						}

					}

					if (interiorNeighbour_flag) {

						switch (dj[k] != 0) {
						case(true):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i, j + dj[k]).vectorIndex);

							index = (dj[k] == 1) ? 1 : 0;

							values.push_back(dj[k] * 0.25 * (arma::as_scalar(m_Psi.row(CellsU(i, j + index).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i - 1, j + index).vectorIndex) * a)) * CellsV(i, j).dy);

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i, j + index).vectorIndex);

							values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j + dj[k]).vectorIndex) * a) * CellsV(i, j).dy);

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsU(i - 1, j + index).vectorIndex);

							values.push_back(dj[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j + dj[k]).vectorIndex) * a) * CellsV(i, j).dy);

							break;
						case(false):

							rowIndices.push_back(eqCounter);
							columnIndices.push_back(CellsV(i + di[k], j).vectorIndex);

							values.push_back(di[k] * (0.5 * arma::as_scalar(m_Psi.row(CellsV(i + di[k], j).vectorIndex) * a) + 0.25 * arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a)) * CellsV(i, j).dx);

							matrixCentralComponent += di[k] * 0.25 * arma::as_scalar(m_Psi.row(CellsV(i + di[k], j).vectorIndex) * a) * CellsV(i, j).dx;

							break;

						}
					}

				}

				rowIndices.push_back(eqCounter);
				columnIndices.push_back(eqCounter);
				values.push_back(matrixCentralComponent);

				matrixCentralComponent = 0.0;


			}

			++eqCounter;

	}

	arma::Row<double> Jrow(a.n_rows, arma::fill::zeros);

	for (int i = 0; i < a.n_rows; ++i) {
		for (int k = 0; k < values.size(); ++k) {

			Jrow(i) += values[k] * m_Psi(columnIndices[k], i);

		}
	}

	return Jrow;
}