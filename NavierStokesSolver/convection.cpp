#include <iostream>
#include <armadillo>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mesh.h"
#include "solver.h"
#include "boundary.h"
#include "ROM.h"


arma::Col<double> solver::N(const arma::Col<double>& vel)  const{

	arma::Col<double> N(m_mesh.getNumU() + m_mesh.getNumV(), arma::fill::zeros);

	const arma::field<cell>& CellsU = m_mesh.getCellsU();
	const arma::field<cell>& CellsV = m_mesh.getCellsV();

	std::vector<int> di = { 1, 0, -1, 0 };
	std::vector<int> dj = { 0, 1, 0, -1 };

	bool interiorNeighbour_flag = true;
	
	//loop over all U unknowns
	for (arma::uword i = m_mesh.getStartIndUy(); i < m_mesh.getEndIndUy(); ++i) {
		for (arma::uword j = m_mesh.getStartIndUx(); j < m_mesh.getEndIndUx(); ++j) {

			//if interior node
			if (!CellsU(i, j).onBoundary) {

				//loop over neighbouring cells
				for (int k = 0; k < 4; ++k) {

					//if neighbour not on left or right boundary calculate neighbour's contribution to convection operator (use of lround probably suboptimal)
					if (!CellsU(i + di[k], j + dj[k]).onBoundary || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'N' || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'S') {
						
						N(CellsU(i, j).vectorIndex) += 0.5 * vel(CellsU(i + di[k], j + dj[k]).vectorIndex) * 0.5 * (
								(dj[k] != 0)	?	dj[k] * (vel(CellsU(i								, j + dj[k]	).vectorIndex) + vel(CellsU(i								, j).vectorIndex)) * CellsU(i, j).dy
												:	di[k] * (vel(CellsV(i + ((di[k] == 1) ? 1 : 0), j - 1).vectorIndex) + vel(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex)) * CellsU(i, j).dx				//di[k] * (vel(CellsV(i + roundl(0.5 * di[k] + 0.5)	, j - 1		).vectorIndex) + vel(CellsV(i + roundl(0.5 * di[k] + 0.5)	, j).vectorIndex)) * CellsU(i, j).dx
							);
					}
					//if neighbour on boundary
					else {

						//switch over all possible boundary conditions
						switch (CellsU(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::PERIODIC_LR):
							N(CellsU(i, j).vectorIndex) += dj[k] * 0.5 * vel(CellsU(i, 0).vectorIndex) * 0.5 * (vel(CellsU(i, 0).vectorIndex) + vel(CellsU(i, j).vectorIndex)) * CellsU(i, j).dy;
							break;
						}

					}

				}

			}
			//if boundary node
			else {

				//loop over neighbours
				for (int k = 0; k < 4; ++k) {

					//deal with upper and lower fluxes
					if (k == 0 || k == 2) {

						//deal with left and right boundaries in separate code
						if (j != 0 && j != m_mesh.getNumCellsX()) {

							auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
								});

							//if flux on upper or lower boundary
							if (it != CellsU(i, j).boundaryData.end()) {

								switch (it->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):
									N(CellsU(i, j).vectorIndex) += di[k] * 0.5 * vel(CellsU(((it->boundaryDir == 'N') ? 0 : (m_mesh.getEndIndUy() - 1)), j).vectorIndex)
										* 0.5 * (vel(CellsV(0, j).vectorIndex) + vel(CellsV(0, j - 1).vectorIndex)) * CellsU(i, j).dx;
									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}
						}
						//if on left or right boundary
						else {

							auto it1 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
								return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
								});

							auto it2 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [](_boundaryData data) {
								return data.boundaryDir == 'E' || data.boundaryDir == 'W';
								});

							//if flux on upper or lower boundary
							if (it1 != CellsU(i, j).boundaryData.end()) {
								
								//switch over left and right boundary conditions
								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):

									//switch over upper and lower boundary conditions
									switch (it1->B_TYPE) {
									case(B_CONDITION::PERIODIC_UL):

										N(CellsU(i,j).vectorIndex) += di[k] * 0.5 * vel(CellsU(((it1->boundaryDir == 'N') ? 0 : (m_mesh.getEndIndUy() - 1)), j).vectorIndex)
											* 0.5 * (vel(CellsV(0, m_mesh.getEndIndVx() - 1).vectorIndex) + vel(CellsV(0, 0).vectorIndex)) * CellsU(i, j).dx;
										break;
									}

									break;
								}

							}
							//if flux not on upper or lower boundary
							else {

								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_LR):
									N(CellsU(i, j).vectorIndex) += di[k] * 0.5 * vel(CellsU(i + di[k], j).vectorIndex)
										* 0.5 * (vel(CellsV(i + ((di[k] == 1) ? 1 : 0), m_mesh.getEndIndVx() - 1).vectorIndex) + vel(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex)) * CellsU(i, j).dx;
									break;
								}

							}

							interiorNeighbour_flag = false;

						}

					}

					//deal with left and right fluxes
					if (k == 1 || k == 3) {

						auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
							});

						//if on left or right boundary
						if (it != CellsU(i, j).boundaryData.end()) {
							
							//switch over possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_LR):
								N(CellsU(i, j).vectorIndex) += dj[k] * 0.5 * vel(CellsU(i, m_mesh.getEndIndUx() - 1).vectorIndex)
									* 0.5 * (vel(CellsU(i, j).vectorIndex) + vel(CellsU(i, m_mesh.getEndIndUx() - 1).vectorIndex)) * CellsU(i, j).dy;
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

									N(CellsU(i, j).vectorIndex) += dj[k] * 0.5 * vel(CellsU(i, 0).vectorIndex) * 0.5 * (vel(CellsU(i, 0).vectorIndex) + vel(CellsU(i, j).vectorIndex)) * CellsU(i, j).dy;

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}
						}
						

					}

					//if neighbour has interior discretization
					if (interiorNeighbour_flag) {
						N(CellsU(i, j).vectorIndex) += 0.5 * vel(CellsU(i + di[k], j + dj[k]).vectorIndex) * 0.5 * (
							(dj[k] != 0)	?	dj[k] * (vel(CellsU(i								, j + dj[k]	).vectorIndex) + vel(CellsU(i								, j).vectorIndex)) * CellsU(i, j).dy
											:	di[k] * (vel(CellsV(i + ((di[k] == 1) ? 1 : 0), j - 1		).vectorIndex) + vel(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex)) * CellsU(i, j).dx
							);
					}

				}

			}

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
						N(CellsV(i, j).vectorIndex) += 0.5 * vel(CellsV(i + di[k], j + dj[k]).vectorIndex) * 0.5 * (
							(di[k] != 0)	?	di[k] * (vel(CellsV(i + di[k]	, j								).vectorIndex) + vel(CellsV(i		, j								).vectorIndex)) * CellsV(i, j).dx
											:	dj[k] * (vel(CellsU(i, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) + vel(CellsU(i - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex)) * CellsV(i, j).dy //dj[k] * (vel(CellsU(i			, j + roundl(0.5 * dj[k] + 0.5)	).vectorIndex) + vel(CellsU(i - 1	, j + roundl(0.5 * dj[k] + 0.5)	).vectorIndex)) * CellsV(i, j).dy
							);
					}
					//if neighbour on boundary
					else {

						//switch over all possible boundary conditions
						switch (CellsV(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
						case(B_CONDITION::PERIODIC_UL):

							N(CellsV(i, j).vectorIndex) += di[k] * 0.5 * vel(CellsV(0, j).vectorIndex) * 0.5 * (vel(CellsV(0, j).vectorIndex) + vel(CellsV(i, j).vectorIndex)) * CellsV(i, j).dx;

							break;
						}

					}

				}

			}
			//if boundary node
			else {

				//loop over neighbours
				for (int k = 0; k < 4; ++k) {

					//deal with upper and lower fluxes
					if (k == 0 || k == 2) {

						auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							});

						//if on upper or lower boundary
						if (it != CellsV(i, j).boundaryData.end()) {

							//switch over all possible boundary conditions
							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_UL):

								N(CellsV(i, j).vectorIndex) += di[k] * 0.5 * vel(CellsV(m_mesh.getEndIndVy() - 1, j).vectorIndex)
									* 0.5 * (vel(CellsV(i, j).vectorIndex) + vel(CellsV(m_mesh.getEndIndVy() - 1, j).vectorIndex)) * CellsV(i, j).dx;

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

									N(CellsV(i, j).vectorIndex) += di[k] * 0.5 * vel(CellsV(0, j).vectorIndex) * 0.5 * (vel(CellsV(0, j).vectorIndex) + vel(CellsV(i, j).vectorIndex)) * CellsV(i, j).dx;

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}

						}

					}

					//deal with left and right fluxes
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

									N(CellsV(i, j).vectorIndex) += dj[k] * 0.5 * vel(CellsV(i, ((it->boundaryDir == 'E') ? 0 : (m_mesh.getEndIndVx() - 1))).vectorIndex)
										* 0.5 * (vel(CellsU(i, 0).vectorIndex) + vel(CellsU(i - 1, 0).vectorIndex)) * CellsV(i, j).dy;

									break;
								}

								interiorNeighbour_flag = false;

							}
							else {

								interiorNeighbour_flag = true;

							}

						}
						//if on upper or lower boundary
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

										N(CellsV(i, j).vectorIndex) += dj[k] * 0.5 * vel(CellsV(i, ((it1->boundaryDir == 'E') ? 0 : (m_mesh.getEndIndVx() - 1))).vectorIndex)
											* 0.5 * (vel(CellsU(m_mesh.getEndIndUy() - 1, 0).vectorIndex) + vel(CellsU(0, 0).vectorIndex)) * CellsV(i, j).dy;

										break;
									}

									break;
								}

							}
							//if flux not on left or right boundary
							else {

								switch (it2->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									N(CellsV(i, j).vectorIndex) += dj[k] * 0.5 * vel(CellsV(i, j + dj[k]).vectorIndex)
										* 0.5 * (vel(CellsU(m_mesh.getEndIndUy() - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) + vel(CellsU(0, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex)) * CellsV(i, j).dy;

									break;
								}

							}

							interiorNeighbour_flag = false;

						}

					}

					
					if (interiorNeighbour_flag) {
						N(CellsV(i, j).vectorIndex) += 0.5 * vel(CellsV(i + di[k], j + dj[k]).vectorIndex) * 0.5 * (
							(di[k] != 0)	?	di[k] * (vel(CellsV(i + di[k]	, j								).vectorIndex) + vel(CellsV(i		, j								).vectorIndex)) * CellsV(i, j).dx
											:	dj[k] * (vel(CellsU(i			, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) + vel(CellsU(i - 1	, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex)) * CellsV(i, j).dy
							);
					}

				}

			}

		}
	}

	//compiler must move
	return N;
}


//WALKTHROUGH THIS CODE VVV ONE MORE TIME

//evaluate a single index with rom scaling coefficients as input
double ROM_Solver::Nindex(const arma::Col<double>& a, arma::uword vecIndex, arma::uword i, arma::uword j) const {

	double N = 0.0;

	const arma::field<cell>& CellsU = m_solver.getMesh().getCellsU();
	const arma::field<cell>& CellsV = m_solver.getMesh().getCellsV();

	std::vector<int> di = { 1, 0, -1, 0 };
	std::vector<int> dj = { 0, 1, 0, -1 };

	bool interiorNeighbour_flag = true;

	//if u node
	if (vecIndex < m_solver.getMesh().getNumU()) {

		double uConvd = 0.0;

		//if interior node
		if (!CellsU(i, j).onBoundary) {

			//loop over neighbouring cells
			for (int k = 0; k < 4; ++k) {

				//if neighbour not on left or right boundary calculate neighbour's contribution to convection operator (use of lround probably suboptimal)
				if (!CellsU(i + di[k], j + dj[k]).onBoundary || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'N' || CellsU(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'S') {

					uConvd = arma::as_scalar(m_Psi.row(CellsU(i + di[k], j + dj[k]).vectorIndex) * a);

					N += 0.5 * uConvd * 0.5 * (
						(dj[k] != 0) ? dj[k] * (	arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a) + 
													uConvd																						) * CellsU(i, j).dy
						: di[k] * (					arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), j - 1).vectorIndex) * a)  + 
													arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex) * a)			) * CellsU(i, j).dx				//di[k] * (vel(CellsV(i + roundl(0.5 * di[k] + 0.5)	, j - 1		).vectorIndex) + vel(CellsV(i + roundl(0.5 * di[k] + 0.5)	, j).vectorIndex)) * CellsU(i, j).dx
						);
				}
				//if neighbour on boundary
				else {

					//switch over all possible boundary conditions
					switch (CellsU(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
					case(B_CONDITION::PERIODIC_LR):
						N += dj[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a) * 0.5 * (arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a)  + arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a)) * CellsU(i, j).dy;
						break;
					}

				}

			}

		}
		//if boundary node
		else {

			//TO DO: VERIFY IMPLEMENTATION

			//loop over neighbours
			for (int k = 0; k < 4; ++k) {

				//deal with upper and lower fluxes
				if (k == 0 || k == 2) {

					//deal with left and right boundaries in separate code
					if (j != 0 && j != m_solver.getMesh().getNumCellsX()) {

						auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							});

						//if flux on upper or lower boundary
						if (it != CellsU(i, j).boundaryData.end()) {

							switch (it->B_TYPE) {
							case(B_CONDITION::PERIODIC_UL):
								N += di[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsU(((it->boundaryDir == 'N') ? 0 : (m_solver.getMesh().getEndIndUy() - 1)), j).vectorIndex) * a)
									* 0.5 * (arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(0, j - 1).vectorIndex) * a)) * CellsU(i, j).dx;
								break;
							}

							interiorNeighbour_flag = false;

						}
						else {

							interiorNeighbour_flag = true;

						}
					}
					//if on left or right boundary
					else {

						auto it1 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
							return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
							});

						auto it2 = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [](_boundaryData data) {
							return data.boundaryDir == 'E' || data.boundaryDir == 'W';
							});

						//if flux on upper or lower boundary
						if (it1 != CellsU(i, j).boundaryData.end()) {

							//switch over left and right boundary conditions
							switch (it2->B_TYPE) {
							case(B_CONDITION::PERIODIC_LR):

								//switch over upper and lower boundary conditions
								switch (it1->B_TYPE) {
								case(B_CONDITION::PERIODIC_UL):

									N += di[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsU(((it1->boundaryDir == 'N') ? 0 : (m_solver.getMesh().getEndIndUy() - 1)), j).vectorIndex) * a)
										* 0.5 * (arma::as_scalar(m_Psi.row(CellsV(0, m_solver.getMesh().getEndIndVx() - 1).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(0, 0).vectorIndex) * a)) * CellsU(i, j).dx;
									break;
								}

								break;
							}

						}
						//if flux not on upper or lower boundary
						else {

							switch (it2->B_TYPE) {
							case(B_CONDITION::PERIODIC_LR):
								N += di[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsU(i + di[k], j).vectorIndex) * a)
									* 0.5 * (arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), m_solver.getMesh().getEndIndVx() - 1).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex) * a)) * CellsU(i, j).dx;
								break;
							}

						}

						interiorNeighbour_flag = false;

					}

				}

				//deal with left and right fluxes
				if (k == 1 || k == 3) {

					auto it = std::find_if(CellsU(i, j).boundaryData.begin(), CellsU(i, j).boundaryData.end(), [k](_boundaryData data) {
						return (k == 1 && data.boundaryDir == 'E') || (k == 3 && data.boundaryDir == 'W');
						});

					//if on left or right boundary
					if (it != CellsU(i, j).boundaryData.end()) {

						//switch over possible boundary conditions
						switch (it->B_TYPE) {
						case(B_CONDITION::PERIODIC_LR):
							N += dj[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsU(i, m_solver.getMesh().getEndIndUx() - 1).vectorIndex) * a)
								* 0.5 * (arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i, m_solver.getMesh().getEndIndUx() - 1).vectorIndex) * a)) * CellsU(i, j).dy;
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

								N += dj[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a) * 0.5 * (arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a)) * CellsU(i, j).dy;

								break;
							}

							interiorNeighbour_flag = false;

						}
						else {

							interiorNeighbour_flag = true;

						}
					}


				}

				//if neighbour has interior discretization
				if (interiorNeighbour_flag) {
					N += 0.5 * arma::as_scalar(m_Psi.row(CellsU(i + di[k], j + dj[k]).vectorIndex) * a) * 0.5 * (
						(dj[k] != 0) ? dj[k] * (arma::as_scalar(m_Psi.row(CellsU(i, j + dj[k]).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i, j).vectorIndex) * a)) * CellsU(i, j).dy
						: di[k] * (arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), j - 1).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i + ((di[k] == 1) ? 1 : 0), j).vectorIndex) * a)) * CellsU(i, j).dx
						);
				}

			}

		}
	}
	//if V node
	else {
		//if interior node
		if (!CellsV(i, j).onBoundary) {

			//loop over neigbouring cells
			for (int k = 0; k < 4; ++k) {

				//if neighbour not on upper or lower boundary calculate neighbour's contribution to convection operator
				if (!CellsV(i + di[k], j + dj[k]).onBoundary || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'E' || CellsV(i + di[k], j + dj[k]).boundaryData[0].boundaryDir == 'W') {
					N += 0.5 * arma::as_scalar(m_Psi.row(CellsV(i + di[k], j + dj[k]).vectorIndex) * a) * 0.5 * (
						(di[k] != 0) ? di[k] * (arma::as_scalar(m_Psi.row(CellsV(i + di[k], j).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a)) * CellsV(i, j).dx
						: dj[k] * (arma::as_scalar(m_Psi.row(CellsU(i, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a)) * CellsV(i, j).dy //dj[k] * (vel(CellsU(i			, j + roundl(0.5 * dj[k] + 0.5)	).vectorIndex) + vel(CellsU(i - 1	, j + roundl(0.5 * dj[k] + 0.5)	).vectorIndex)) * CellsV(i, j).dy
						);
				}
				//if neighbour on boundary
				else {

					//switch over all possible boundary conditions
					switch (CellsV(i + di[k], j + dj[k]).boundaryData[0].B_TYPE) {
					case(B_CONDITION::PERIODIC_UL):

						N += di[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a) * 0.5 * (arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a)) * CellsV(i, j).dx;

						break;
					}

				}

			}

		}
		//if boundary node
		else {

			//loop over neighbours
			for (int k = 0; k < 4; ++k) {

				//deal with upper and lower fluxes
				if (k == 0 || k == 2) {

					auto it = std::find_if(CellsV(i, j).boundaryData.begin(), CellsV(i, j).boundaryData.end(), [k](_boundaryData data) {
						return (k == 0 && data.boundaryDir == 'N') || (k == 2 && data.boundaryDir == 'S');
						});

					//if on upper or lower boundary
					if (it != CellsV(i, j).boundaryData.end()) {

						//switch over all possible boundary conditions
						switch (it->B_TYPE) {
						case(B_CONDITION::PERIODIC_UL):

							N += di[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsV(m_solver.getMesh().getEndIndVy() - 1, j).vectorIndex) * a)
								* 0.5 * (arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(m_solver.getMesh().getEndIndVy() - 1, j).vectorIndex) * a)) * CellsV(i, j).dx;

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

								N += di[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a) * 0.5 * (arma::as_scalar(m_Psi.row(CellsV(0, j).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a)) * CellsV(i, j).dx;

								break;
							}

							interiorNeighbour_flag = false;

						}
						else {

							interiorNeighbour_flag = true;

						}

					}

				}

				//deal with left and right fluxes
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

								N += dj[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsV(i, ((it->boundaryDir == 'E') ? 0 : (m_solver.getMesh().getEndIndVx() - 1))).vectorIndex) * a)
									* 0.5 * (arma::as_scalar(m_Psi.row(CellsU(i, 0).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i - 1, 0).vectorIndex) * a)) * CellsV(i, j).dy;

								break;
							}

							interiorNeighbour_flag = false;

						}
						else {

							interiorNeighbour_flag = true;

						}

					}
					//if on upper or lower boundary
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

									N += dj[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsV(i, ((it1->boundaryDir == 'E') ? 0 : (m_solver.getMesh().getEndIndVx() - 1))).vectorIndex) * a)
										* 0.5 * (arma::as_scalar(m_Psi.row(CellsU(m_solver.getMesh().getEndIndUy() - 1, 0).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(0, 0).vectorIndex) * a)) * CellsV(i, j).dy;

									break;
								}

								break;
							}

						}
						//if flux not on left or right boundary
						else {

							switch (it2->B_TYPE) {
							case(B_CONDITION::PERIODIC_UL):

								N += dj[k] * 0.5 * arma::as_scalar(m_Psi.row(CellsV(i, j + dj[k]).vectorIndex) * a)
									* 0.5 * (arma::as_scalar(m_Psi.row(CellsU(m_solver.getMesh().getEndIndUy() - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(0, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a)) * CellsV(i, j).dy;

								break;
							}

						}

						interiorNeighbour_flag = false;

					}

				}


				if (interiorNeighbour_flag) {
					N += 0.5 * arma::as_scalar(m_Psi.row(CellsV(i + di[k], j + dj[k]).vectorIndex) * a) * 0.5 * (
						(di[k] != 0) ? di[k] * (arma::as_scalar(m_Psi.row(CellsV(i + di[k], j).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsV(i, j).vectorIndex) * a)) * CellsV(i, j).dx
						: dj[k] * (arma::as_scalar(m_Psi.row(CellsU(i, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a) + arma::as_scalar(m_Psi.row(CellsU(i - 1, j + ((dj[k] == 1) ? 1 : 0)).vectorIndex) * a)) * CellsV(i, j).dy
						);
				}

			}

		}
	}

	//compiler must move
	return N;
}

