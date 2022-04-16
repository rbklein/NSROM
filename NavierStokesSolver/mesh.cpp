#include <iostream>
#include <armadillo>

#include "mesh.h"
#include "boundary.h"

// process boundary condition information from solver object into mesh cells (will need update for less trivial boundary conditions requiring function evaluations)

void mesh::processBoundary(B_CONDITION bcUp, B_CONDITION bcRight, B_CONDITION bcLower, B_CONDITION bcLeft) {
	int nCellsUx = m_nCellsX + 1;
	int nCellsUy = m_nCellsY;

	int nCellsVx = m_nCellsX;
	int nCellsVy = m_nCellsY + 1;

	for (arma::uword i = 0; i < nCellsUy; ++i) {
		for (arma::uword j = 0; j < nCellsUx; ++j) {

			// if on lower boundary
			if (i == 0) {
				m_uCells(i, j).onBoundary = true;
				m_uCells(i, j).boundaryData.push_back({ 'S', bcLower });
			}

			// if on upper boundary
			if (i == (nCellsUy - 1)) {
				m_uCells(i, j).onBoundary = true;
				m_uCells(i, j).boundaryData.push_back({ 'N', bcUp });
			}

			// if on left boundary
			if (j == 0) {
				m_uCells(i, j).onBoundary = true;
				m_uCells(i, j).boundaryData.push_back({ 'W', bcLeft });
			}

			// if on right boundary
			if (j == (nCellsUx - 1)) {
				m_uCells(i, j).onBoundary = true;
				m_uCells(i, j).boundaryData.push_back({ 'E', bcRight });
			}

			// if in interior
			if (i != 0 && i != (nCellsUy - 1) && j != 0 && j != (nCellsUx - 1)) {
				m_uCells(i, j).onBoundary = false;
				m_uCells(i, j).boundaryData.push_back({ 'C', B_CONDITION::INTERIOR });
			}
		}
	}

	for (arma::uword i = 0; i < nCellsVy; ++i) {
		for (arma::uword j = 0; j < nCellsVx; ++j) {

			// if on lower boundary
			if (i == 0) {
				m_vCells(i, j).onBoundary = true;
				m_vCells(i, j).boundaryData.push_back({ 'S', bcLower });
			}

			// if on upper boundary
			if (i == (nCellsVy - 1)) {
				m_vCells(i, j).onBoundary = true;
				m_vCells(i, j).boundaryData.push_back({ 'N', bcUp });
			}

			// if on left boundary
			if (j == 0) {
				m_vCells(i, j).onBoundary = true;
				m_vCells(i, j).boundaryData.push_back({ 'W', bcLeft });
			}

			// if on right boundary
			if (j == (nCellsVx - 1)) {
				m_vCells(i, j).onBoundary = true;
				m_vCells(i, j).boundaryData.push_back({ 'E', bcRight });
			}

			// if in interior
			if (i != 0 && i != (nCellsVy - 1) && j != 0 && j != (nCellsVx - 1)) {
				m_vCells(i, j).onBoundary = false;
				m_vCells(i, j).boundaryData.push_back({ 'C', B_CONDITION::INTERIOR });
			}
		}
	}

	for (arma::uword i = 0; i < m_nCellsY; ++i) {
		for (arma::uword j = 0; j < m_nCellsX; ++j) {
			// if on lower boundary
			if (i == 0) {
				m_pCells(i, j).onBoundary = true;
				m_pCells(i, j).boundaryData.push_back({ 'S', bcLower });
			}

			// if on upper boundary
			if (i == (m_nCellsY - 1)) {
				m_pCells(i, j).onBoundary = true;
				m_pCells(i, j).boundaryData.push_back({ 'N', bcUp });
			}

			// if on left boundary
			if (j == 0) {
				m_pCells(i, j).onBoundary = true;
				m_pCells(i, j).boundaryData.push_back({ 'W', bcLeft });
			}

			// if on right boundary
			if (j == (m_nCellsX - 1)) {
				m_pCells(i, j).onBoundary = true;
				m_pCells(i, j).boundaryData.push_back({ 'E', bcRight });
			}

			// if in interior
			if (i != 0 && i != (m_nCellsY - 1) && j != 0 && j != (m_nCellsY - 1)) {
				m_pCells(i, j).onBoundary = false;
				m_pCells(i, j).boundaryData.push_back({ 'C', B_CONDITION::INTERIOR });
			}
		}
	}

	//initiate indices to loop over

	m_startIndUx = 0;
	m_endIndUx = nCellsUx;
	m_startIndUy = 0;
	m_endIndUy = nCellsUy;

	m_startIndVx = 0;
	m_endIndVx = nCellsVx;
	m_startIndVy = 0;
	m_endIndVy = nCellsVy;

	if (bcUp == B_CONDITION::PERIODIC_UL || bcLower == B_CONDITION::PERIODIC_UL) {
		--m_endIndVy;
		--nCellsVy;

		//adapt cells sizes as if domain were periodic
		for (int j = 0; j < nCellsVx; ++j) {
			m_vCells(0, j).dy = m_vCells(0, j).dy + m_vCells(nCellsVy, j).dy;
		}
	}

	if (bcRight == B_CONDITION::PERIODIC_LR || bcLeft == B_CONDITION::PERIODIC_LR) {
		--m_endIndUx;
		--nCellsUx;

		//adapt cells sizes as if domain were periodic
		for (int i = 0; i < nCellsUy; ++i) {
			m_uCells(i, 0).dx = m_uCells(i, 0).dx + m_uCells(i, nCellsUx).dx;
		}
	}

	if (bcUp == B_CONDITION::NO_SLIP) {
		--m_endIndVy;
		--nCellsVy;
	}

	if (bcLower == B_CONDITION::NO_SLIP) {
		++m_startIndVy;
		--nCellsVy;
	}

	if (bcRight == B_CONDITION::NO_SLIP) {
		--m_endIndUx;
		--nCellsUx;
	}

	if (bcLeft == B_CONDITION::NO_SLIP) {
		++m_startIndUx;
		--nCellsUx;
	}

	arma::uword count = 0;

	for (arma::uword i = m_startIndUy; i < m_endIndUy; ++i) {
		for (arma::uword j = m_startIndUx; j < m_endIndUx; ++j) {
			m_uCells(i, j).vectorIndex = count;
			++count;
		}
	}

	m_nU = count;

	for (arma::uword i = m_startIndVy; i < m_endIndVy; ++i) {
		for (arma::uword j = m_startIndVx; j < m_endIndVx; ++j) {
			m_vCells(i, j).vectorIndex = count;
			++count;
		}
	}

	m_nV = count - m_nU;

	count = 0;

	for (arma::uword i = 0; i < m_nCellsY; ++i) {
		for (arma::uword j = 0; j < m_nCellsX; ++j) {
			m_pCells(i, j).vectorIndex = count;
			++count;
		}
	}

	
	
}

arma::uword mesh::getStartIndUx() const {
	return m_startIndUx;
}

arma::uword mesh::getStartIndUy() const {
	return m_startIndUy;
}

arma::uword mesh::getEndIndUx() const {
	return m_endIndUx;
}

arma::uword mesh::getEndIndUy() const {
	return m_endIndUy;
}

arma::uword mesh::getStartIndVx() const {
	return m_startIndVx;
}

arma::uword mesh::getStartIndVy() const {
	return m_startIndVy;
}

arma::uword mesh::getEndIndVx() const {
	return m_endIndVx;
}

arma::uword mesh::getEndIndVy() const {
	return m_endIndVy;
}


int mesh::getNumU() const {
	return m_nU;
}

int mesh::getNumV() const {
	return m_nV;
}

arma::field<cell> mesh::constructCellsU() {
	int nCellsUx = m_nCellsX + 1;
	int nCellsUy = m_nCellsY;

	double dx = m_lengthX / m_nCellsX;
	double dy = m_lengthY / m_nCellsY;

	arma::field<cell> uCells(nCellsUy, nCellsUx);

	for (arma::uword i = 0; i < nCellsUy; ++i) {
		for (arma::uword j = 0; j < nCellsUx; ++j) {
			uCells(i, j).x = j * dx;
			uCells(i, j).y = i * dy + 0.5 * dy;
			
			uCells(i, j).dx = (j == 0 || j == (nCellsUx - 1)) ? 0.5 * dx : dx;
			uCells(i, j).dy = dy;
		}
	}

	return uCells; //is this moved?
}

arma::field<cell> mesh::constructCellsV() {
	int nCellsVx = m_nCellsX;
	int nCellsVy = m_nCellsY + 1;

	double dx = m_lengthX / m_nCellsX;
	double dy = m_lengthY / m_nCellsY;

	arma::field<cell> vCells(nCellsVy, nCellsVx);

	for (arma::uword i = 0; i < nCellsVy; ++i) {
		for (arma::uword j = 0; j < nCellsVx; ++j) {
			vCells(i, j).x = j * dx + 0.5 * dx;
			vCells(i, j).y = i * dy;

			vCells(i, j).dx = dx;
			vCells(i, j).dy = (i == 0 || i == (nCellsVy - 1)) ? 0.5 * dy : dy;
		}
	}

	return vCells; //is this moved?
}

arma::field<cell> mesh::constructCellsP() {
	int nCellsPx = m_nCellsX;
	int nCellsPy = m_nCellsY;

	double dx = m_lengthX / m_nCellsX;
	double dy = m_lengthY / m_nCellsY;

	arma::field<cell> pCells(nCellsPy, nCellsPx);

	for (arma::uword i = 0; i < nCellsPy; ++i) {
		for (arma::uword j = 0; j < nCellsPx; ++j) {
			pCells(i, j).x = j * dx + 0.5 * dx;
			pCells(i, j).y = i * dy + 0.5 * dy;

			pCells(i, j).dx = dx;
			pCells(i, j).dy = dy;
		}
	}

	return pCells; //is this moved?
}

const arma::field<cell>& mesh::getCellsU() const {
	return this->m_uCells;
}

const arma::field<cell>& mesh::getCellsV() const {
	return this->m_vCells;
}

const arma::field<cell>& mesh::getCellsP() const {
	return this->m_pCells;
}

int mesh::getNumCellsX() const {
	return m_nCellsX;
}

int mesh::getNumCellsY() const {
	return m_nCellsY;
}

double mesh::getLengthX() const {
	return m_lengthX;
}

double mesh::getLengthY() const {
	return m_lengthY;
}