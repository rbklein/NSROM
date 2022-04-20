#include <iostream>
#include <armadillo>
#include <cmath>

#include "mesh.h"
#include "solver.h"
#include "data.h"
#include "ROM.h"

//CHECK THIS: MODES ARE NOT ALWAYS DIVERGENCE FREE!!!

//make modes omega-orthogonal and momentum conserving (divergence-free automatically) 
//IMPORTANT: realize -> momentum conserving modes are also divergence-free!!
void ROM_Solver::setupBasis() {

	const arma::field<cell>& CellsU = m_solver.getMesh().getCellsU();
	const arma::field<cell>& CellsV = m_solver.getMesh().getCellsV();

	arma::Col<double> Eu = arma::zeros(m_solver.getMesh().getNumU() + m_solver.getMesh().getNumV());
	arma::Col<double> Ev = arma::zeros(m_solver.getMesh().getNumU() + m_solver.getMesh().getNumV());

	//setup momentum conserving modes
	for (arma::uword i = m_solver.getMesh().getStartIndUy(); i < m_solver.getMesh().getEndIndUy(); ++i) {
		for (arma::uword j = m_solver.getMesh().getStartIndUx(); j < m_solver.getMesh().getEndIndUx(); ++j) {
			Eu(CellsU(i, j).vectorIndex) = 1.0;
		}
	}

	for (arma::uword i = m_solver.getMesh().getStartIndVy(); i < m_solver.getMesh().getEndIndVy(); ++i) {
		for (arma::uword j = m_solver.getMesh().getStartIndVx(); j < m_solver.getMesh().getEndIndVx(); ++j) {
			Ev(CellsV(i, j).vectorIndex) = 1.0;
		}
	}

	//normalize momentum conserving modes in omega-norm
	Eu = 1.0 / sqrt(arma::as_scalar(Eu.t() * m_solver.Om() * Eu)) * Eu;
	Ev = 1.0 / sqrt(arma::as_scalar(Ev.t() * m_solver.Om() * Ev)) * Ev;

	arma::Mat<double> E = arma::join_rows(Eu, Ev);

	arma::Mat<double> PsiFull, _;
	arma::Col<double> singularValues;

	//get snapshot data
	arma::Mat<double> scaledSnapshotData = m_dataCollector.getDataMatrix();

	//scale snapshots for omega-orthogonality
	scaledSnapshotData = arma::sqrt(m_solver.Om()) * scaledSnapshotData; 

	//subtract omega-weighted projections of snapshots on E
	scaledSnapshotData = scaledSnapshotData - E * E.t() * m_solver.Om() * scaledSnapshotData;

	//perform svd of scaled snapshots
	arma::svd_econ(PsiFull, singularValues, _, scaledSnapshotData);

	//add momentum conserving modes
	m_Psi.insert_cols(0, E);

	//ensures omega-orthogonality
	for (arma::uword i = 0; i < (m_numModesPOD - 2); ++i) {
		m_Psi.insert_cols(m_Psi.n_cols, arma::sqrt(m_solver.OmInv()) * PsiFull.col(i));
	}

	std::cout << (m_solver.M() * m_Psi).max() << " " << (m_solver.M() * m_Psi).min() << std::endl;

}

//precompute reduced diffusion operator (future: implement tensor decomposition)
void ROM_Solver::precomputeOperators() {

	m_Dr = m_Psi.t() * m_solver.D() * m_Psi;

}

arma::Col<double> ROM_Solver::calculateIC(const arma::Col<double>& vel) const {
	return m_Psi.t() * m_solver.Om() * vel;
}

const arma::Mat<double>& ROM_Solver::Dr() const {
	return m_Dr;
}

const arma::Mat<double>& ROM_Solver::Psi() const {
	return m_Psi;
}

arma::Col<double> ROM_Solver::Nr(const arma::Col<double>& a) const {
	return m_hyperReduction.Nrh(a, *this);
}

double ROM_Solver::nu() const {
	return m_solver.nu();
}

const solver& ROM_Solver::getSolver() const {
	return m_solver;
}