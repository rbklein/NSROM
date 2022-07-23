#ifndef H_MESH
#define H_MESH

#include <iostream>
#include <armadillo>
#include <vector>

#include "boundary.h"

struct _boundaryData {
	char boundaryDir;    // C (interior nodes), N, E, S, W
	B_CONDITION B_TYPE;
};

//stores cell information
struct cell {
	double x, y, dx, dy;

	bool onBoundary;
	std::vector<_boundaryData> boundaryData;    // C (interior nodes), N, E, S, W
	
	arma::uword vectorIndex;
};


// Sets up a mesh on a rectangular domain
class mesh {
private:
	int m_nCellsX, m_nCellsY;

	int m_nU, m_nV;

	double m_lengthX, m_lengthY;

	arma::field<cell> m_uCells, m_vCells, m_pCells;

	arma::uword m_startIndUx, m_endIndUx, m_startIndUy, m_endIndUy;
	arma::uword m_startIndVx, m_endIndVx, m_startIndVy, m_endIndVy;

public:

	mesh(){}

	mesh(int nCellsX, int nCellsY, double lengthX, double lengthY) 
			:	m_nCellsX(nCellsX), m_nCellsY(nCellsY), m_lengthX(lengthX), m_lengthY(lengthY),
				m_uCells(), m_vCells(), m_pCells()
	{

		m_uCells		= constructCellsU();
		m_vCells		= constructCellsV();
		m_pCells		= constructCellsP();

	}

	const arma::field<cell>& getCellsU() const;
	const arma::field<cell>& getCellsV() const;
	const arma::field<cell>& getCellsP() const;

	int getNumCellsX() const;
	int getNumCellsY() const;

	int getNumU() const;
	int getNumV() const;

	arma::uword getStartIndUx() const;
	arma::uword getStartIndUy() const;
	arma::uword getEndIndUx() const;
	arma::uword getEndIndUy() const;
	arma::uword getStartIndVx() const;
	arma::uword getStartIndVy() const;
	arma::uword getEndIndVx() const;
	arma::uword getEndIndVy() const;

	double getLengthX() const;
	double getLengthY() const;

	void processBoundary(B_CONDITION bcUp, B_CONDITION bcRight, B_CONDITION bcLower, B_CONDITION bcLeft);

private:
	arma::field<cell> constructCellsU();
	arma::field<cell> constructCellsV();
	arma::field<cell> constructCellsP();
};

#endif