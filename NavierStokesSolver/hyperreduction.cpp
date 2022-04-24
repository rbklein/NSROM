#include <iostream>
#include <armadillo>

#include "mesh.h"
#include "solver.h"
#include "ROM.h"

HYPER_REDUCTION_METHOD Base_hyperReduction::getType() const {
	return m_method;
}

arma::Col<double> noHyperReduction::Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {
	return rom_solver.Psi().t() * rom_solver.getSolver().N(rom_solver.Psi() * a);
}

void noHyperReduction::initialize(const ROM_Solver& rom_solver) {
	//no initialization steps necessary
}

void DEIM::setupMeasurementSpace() {

	//get reference to data matrix
	const arma::Mat<double> operatorMatrix = m_collector.getOperatorMatrix();

	//declare singular vectors and singular values
	arma::Mat<double> MFull, _;
	arma::Col<double> singularValues;

	//perform svd of snapshots
	arma::svd_econ(MFull, singularValues, _, operatorMatrix);

	singularValues.save("deim_sing_vals.txt", arma::raw_ascii);

	//get amount of unknowns
	arma::uword n = operatorMatrix.n_rows;

	//set size of measurement space matrix
	m_P = arma::SpMat<double>(n, m_numModes);

	//find max index of first data matrix column
	arma::uword ind = operatorMatrix.col(0).index_max();

	m_indsP.push_back(ind);

	//update measurement space
	m_P(ind, 0) = 1.0;

	//declare c vector
	arma::Col<double> c;

	for (arma::uword i = 1; i < m_numModes; ++i) {
		//equate next column and previous modes in measurement space (not sure why this does not solve for m_numModes components, but it seems to work)
		c = arma::solve(m_P.t() * MFull.cols(0, i - 1), (m_P.t() * MFull.col(i)).as_col());

		//find index of max absolute difference in FOM space
		ind = arma::abs(MFull.col(i) - MFull.cols(0, i - 1) * c).index_max();
		m_indsP.push_back(ind);

		//update measurement space
		m_P(ind, i) = 1.0;
	}

	//define deim modes
	m_M = MFull.cols(0, m_numModes - 1);

	//lu decompose deim modes in measurement space for fast inversion
	arma::lu(m_PTM_L, m_PTM_U, m_PTM_perm, m_P.t() * m_M);

}

arma::Col<double> DEIM::Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Col<double> c(m_numModes);
	
	arma::Col<double> PTN(m_numModes);

	for (int i = 0; i < m_numModes; ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);
	}

	c = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * PTN));

	return m_PsiTM * c;
}

void DEIM::initialize(const ROM_Solver& rom_solver) {

	for (arma::uword ind : m_indsP) {
		m_gridIndsP.push_back(rom_solver.getSolver().vectorToGridIndex(ind));
	}

	m_PsiTM = rom_solver.Psi().t() * m_M;
}





void SPDEIM::setupMeasurementSpace() {

	//get reference to data matrix
	const arma::Mat<double>& operatorMatrix = m_collector.getOperatorMatrix();

	//declare singular vectors and singular values
	arma::Mat<double> MFull, _;
	arma::Col<double> singularValues;

	//perform svd of snapshots
	arma::svd_econ(MFull, singularValues, _, operatorMatrix);

	singularValues.save("deim_sing_vals.txt", arma::raw_ascii);

	//get amount of unknowns
	arma::uword n = operatorMatrix.n_rows;

	//set size of measurement space matrix
	m_P = arma::SpMat<double>(n, m_numModes - 1);

	//find max index of first data matrix column
	arma::uword ind = operatorMatrix.col(0).index_max();

	m_indsP.push_back(ind);

	//update measurement space
	m_P(ind, 0) = 1.0;

	//declare c vector
	arma::Col<double> c;

	//only have (m_numModes - 1) number of measurement points
	for (arma::uword i = 1; i < (m_numModes - 1); ++i) {
		//equate next column and previous modes in measurement space (not sure why this does not solve for m_numModes components, but it seems to work)
		c = arma::solve(m_P.t() * MFull.cols(0, i - 1), (m_P.t() * MFull.col(i)).as_col());

		//find index of max absolute difference in FOM space
		ind = arma::abs(MFull.col(i) - MFull.cols(0, i - 1) * c).index_max();
		m_indsP.push_back(ind);

		//update measurement space
		m_P(ind, i) = 1.0;
	}

	//define deim modes
	m_M = MFull.cols(0, m_numModes - 1);

	m_PTM = m_P.t() * m_M;

	//lu decompose deim modes in measurement space for fast inversion
	arma::lu(m_Mp_L, m_Mp_U, m_Mp_perm, m_PTM.cols(0, m_numModes - 2));

	m_MpiMm = arma::solve(arma::trimatu(m_Mp_U), arma::solve(arma::trimatl(m_Mp_L), m_Mp_perm * m_PTM.col(m_numModes-1)));

}

arma::Col<double> SPDEIM::Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Col<double> c(m_numModes);

	arma::Col<double> MpiPTN;

	arma::Col<double> ra(m_numModes - 1);

	arma::Col<double> PTN(m_numModes - 1);

	arma::Row<double> aTPsiTM = a.t() * m_PsiTM;

	for (int i = 0; i < (m_numModes - 1); ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);

		ra(i) = - aTPsiTM(i) / aTPsiTM(m_numModes - 1);
	}

	MpiPTN = arma::solve(arma::trimatu(m_Mp_U), arma::solve(arma::trimatl(m_Mp_L), m_Mp_perm * PTN));

	c.rows(0, m_numModes - 2) = MpiPTN - arma::as_scalar(ra.t() * MpiPTN) / (1.0 + arma::as_scalar(ra.t() * m_MpiMm)) * m_MpiMm;

	c.row(m_numModes - 1) = arma::as_scalar(ra.t() * c.rows(0, m_numModes - 2));

	return m_PsiTM * c;
}


void SPDEIM::initialize(const ROM_Solver& rom_solver) {

	for (arma::uword ind : m_indsP) {
		m_gridIndsP.push_back(rom_solver.getSolver().vectorToGridIndex(ind));
	}

	m_PsiTM = rom_solver.Psi().t() * m_M;
}
