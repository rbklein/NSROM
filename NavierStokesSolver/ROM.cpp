#include <iostream>
#include <armadillo>
#include <cmath>

#include "mesh.h"
#include "solver.h"
#include "data.h"
#include "ROM.h"

//#define LOAD_DATA

void ROM_Solver::setupBasis() {

#ifndef LOAD_DATA

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
	Eu = (1.0 / sqrt(arma::as_scalar(Eu.t() * m_solver.Om() * Eu))) * Eu;
	Ev = (1.0 / sqrt(arma::as_scalar(Ev.t() * m_solver.Om() * Ev))) * Ev;

	arma::Mat<double> E = arma::join_rows(Eu, Ev);

	arma::Mat<double> PsiFull, _;
	arma::Col<double> singularValues;

	//get snapshot data
	arma::Mat<double> scaledSnapshotData = m_dataCollector.getDataMatrix();

	//arma::Mat<double> scaledSnapshotData;
	//scaledSnapshotData.load("solution_snapshots_slr");  //.load("solution_snapshots_2dturb_" + std::to_string(m_datasetIndex));  //

	//subtract omega-weighted projections of snapshots on E
	arma::Col<double> ee;

	for (int j = 0; j < scaledSnapshotData.n_cols; ++j) {

		ee = E.t() * m_solver.Om() * scaledSnapshotData.col(j);

		//subtract omega-weighted projections of snapshots on E
		scaledSnapshotData.col(j) = scaledSnapshotData.col(j) - E * ee;
	}


	//scale snapshots for omega-orthogonality
	scaledSnapshotData = arma::sqrt(m_solver.Om()) * scaledSnapshotData; 

	//perform svd of scaled snapshots
	arma::svd_econ(PsiFull, singularValues, _, scaledSnapshotData, "left");

	singularValues.save("pod_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::sum(singularValues.rows(0, m_numModesPOD - 3)) / arma::sum(singularValues);

	std::cout << "POD RIC: " << m_RIC * 100.0 << "% " << std::endl;

	//ensures omega-orthogonality
	for (arma::uword i = 0; i < (m_numModesPOD - 2); ++i) {
		m_Psi.insert_cols(m_Psi.n_cols, arma::sqrt(m_solver.OmInv()) * PsiFull.col(i));
	}

	//add momentum conserving modes
	m_Psi.insert_cols(0, E);

	if (m_savePhi) {

		PsiFull = arma::sqrt(m_solver.OmInv()) * PsiFull;
		PsiFull.insert_cols(0, E);
		PsiFull.save("PsiFull.bin", arma::arma_binary);

	}

#else

	arma::Mat<double> PsiFull;
	PsiFull.load("PsiFull.bin", arma::arma_binary);
	
	arma::Col<double> singularValues;
	singularValues.load("pod_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::sum(singularValues.rows(0, m_numModesPOD - 3)) / arma::sum(singularValues);

	std::cout << "POD RIC: " << m_RIC * 100.0 << "% " << std::endl;

	m_Psi = PsiFull.cols(0, m_numModesPOD - 1);


#endif


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

arma::Mat<double> ROM_Solver::Jr(const arma::Col<double>& a) const {
	return m_hyperReduction.Jrh(a, *this);
}

double ROM_Solver::nu() const {
	return m_solver.nu();
}

const solver& ROM_Solver::getSolver() const {
	return m_solver;
}

const Base_hyperReduction& ROM_Solver::getHyperReduction() const {
	return m_hyperReduction;
}
