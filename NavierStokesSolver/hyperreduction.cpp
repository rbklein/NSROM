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

arma::Mat<double> noHyperReduction::Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {
	return arma::Mat<double>(rom_solver.Psi().t() * rom_solver.getSolver().J(rom_solver.Psi() * a));
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
	arma::svd_econ(MFull, singularValues, _, operatorMatrix, "left", "std");

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
	

	//c = arma::solve(m_P.t() * m_M, m_P.t() * rom_solver.getSolver().N(rom_solver.Psi() * a));

	std::cout << "diff operators: " << arma::norm(m_M * c - rom_solver.getSolver().N(rom_solver.Psi() * a), "inf") << std::endl;

	return m_PsiTM * c;
}

void DEIM::initialize(const ROM_Solver& rom_solver) {

	arma::Mat<double> modes;

	for (int k = 0; k < m_numModes; ++k) {
		modes = arma::join_rows(modes, rom_solver.getSolver().interpolateVelocity(m_M.col(k)));
	}

	modes.save("deim_modes.txt", arma::raw_ascii);

	for (arma::uword ind : m_indsP) {
		m_gridIndsP.push_back(rom_solver.getSolver().vectorToGridIndex(ind));
	}

	m_PsiTM = rom_solver.Psi().t() * m_M;
}

arma::Mat<double> DEIM::Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Mat<double> PTJPsi;

	for (int m = 0; m < m_numModes; ++m) {
		PTJPsi = arma::join_cols(PTJPsi, rom_solver.Jindex(a, m_indsP[m], m_gridIndsP[m].first, m_gridIndsP[m].second));
	}

	return m_PsiTM * arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * PTJPsi));
}



void SPDEIM::setupMeasurementSpace() {

	//get reference to data matrix
	const arma::Mat<double>& operatorMatrix = m_collector.getOperatorMatrix();

	//declare singular vectors and singular values
	arma::Mat<double> MFull, _;
	arma::Col<double> singularValues;

	//perform svd of snapshots
	arma::svd_econ(MFull, singularValues, _, operatorMatrix, "left", "std");

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

	//std::cout << "diff operators: " << arma::norm(m_M * c - rom_solver.getSolver().N(rom_solver.Psi() * a), "inf") << std::endl;

	return m_PsiTM * c;
}

arma::Mat<double> SPDEIM::Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {
	
	//std::cout << rom_solver.Psi().t() * m_M << std::endl;

	arma::Mat<double> output;

	arma::Col<double> ra(m_numModes - 1);

	arma::Row<double> aTPsiTM = a.t() * m_PsiTM;

	arma::Col<double> PTN(m_numModes - 1);

	for (int i = 0; i < (m_numModes - 1); ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);

		ra(i) = -aTPsiTM(i) / aTPsiTM(m_numModes - 1);
	}

	double scalarTerm = 1.0 + arma::as_scalar(ra.t() * m_MpiMm);

	arma::Mat<double> drda = (1.0 / aTPsiTM.back()) * (-m_PsiTM.cols(0, m_numModes - 2).t() + ra * m_PsiTM.col(m_numModes - 1).t());

	arma::Col<double> MpiPTN = arma::solve(arma::trimatu(m_Mp_U), arma::solve(arma::trimatl(m_Mp_L), m_Mp_perm * PTN));

	arma::Mat<double> PTJPsi;

	for (int m = 0; m < (m_numModes - 1); ++m) {
		PTJPsi = arma::join_cols(PTJPsi, rom_solver.Jindex(a, m_indsP[m], m_gridIndsP[m].first, m_gridIndsP[m].second));
	}

	arma::Mat<double> MpiPTJ = arma::solve(arma::trimatu(m_Mp_U), arma::solve(arma::trimatl(m_Mp_L), m_Mp_perm * PTJPsi));

	arma::Mat<double> gradient = (1.0 / scalarTerm) * (drda.t() * MpiPTN + MpiPTJ.t() * ra) - arma::as_scalar(ra.t() * MpiPTN) / (scalarTerm * scalarTerm) * drda.t() * m_MpiMm;

	output = MpiPTJ - m_MpiMm * gradient.t();

	arma::Mat<double> gradientCm = (drda.t() * (MpiPTN - arma::as_scalar(ra.t() * MpiPTN) / (1.0 + arma::as_scalar(ra.t() * m_MpiMm)) * m_MpiMm) + output.t() * ra);

	output = m_PsiTM * arma::join_cols(output, gradientCm.t());

	//std::cout << "CONDITION NUMBER: " << arma::cond(output) << std::endl;

	return  output;
}

void SPDEIM::initialize(const ROM_Solver& rom_solver) {

	arma::Mat<double> modes;

	for (int k = 0; k < m_numModes; ++k) {
		modes = arma::join_rows(modes, rom_solver.getSolver().interpolateVelocity(m_M.col(k)));
	}

	modes.save("deim_modes.txt", arma::raw_ascii);

	for (arma::uword ind : m_indsP) {
		m_gridIndsP.push_back(rom_solver.getSolver().vectorToGridIndex(ind));
	}

	m_PsiTM = rom_solver.Psi().t() * m_M;
}





void LSDEIM::setupMeasurementSpace() {

	//get reference to data matrix
	const arma::Mat<double> operatorMatrix = m_collector.getOperatorMatrix();

	//declare singular vectors and singular values
	arma::Mat<double> MFull, _;
	arma::Col<double> singularValues;

	//perform svd of snapshots
	arma::svd_econ(MFull, singularValues, _, operatorMatrix, "left", "std");

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

	std::vector<arma::uword> lol = { 10, 18 };
	std::vector<arma::uword> lmao = { 9965, 18899 };

	for (arma::uword i = 1; i < (m_numModes); ++i) {
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
	arma::lu(m_PTM_L, m_PTM_U, m_PTM_perm, 2.0 * (m_P.t() * m_M).t() * (m_P.t() * m_M));

	m_PTM = m_P.t() * m_M;

	m_M1 = 2.0 * arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * m_PTM.t()));

}

arma::Col<double> LSDEIM::Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {


	arma::Col<double> c(m_numModes);

	arma::Col<double> PTN(m_numModes);

	for (int i = 0; i < m_numModes; ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);
	}

	//c = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * PTN));
	
	arma::Col<double> PTMPTN = 2.0 * m_PTM.t() * PTN;

	arma::Col<double> B = (a.t() * m_PsiTM).t();

	arma::Col<double> AiPTMPTN = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * PTMPTN));

	arma::Col<double> AiB = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * B));

	c = AiPTMPTN - arma::as_scalar((B.t() * AiPTMPTN) / (B.t() * AiB)) * AiB;

	//std::cout << "diff operators: " << arma::norm(m_M * c - rom_solver.getSolver().N(rom_solver.Psi() * a), "inf") << std::endl;

	return m_PsiTM * c;
}

void LSDEIM::initialize(const ROM_Solver& rom_solver) {

	arma::Mat<double> modes;

	for (int k = 0; k < m_numModes; ++k) {
		modes = arma::join_rows(modes, rom_solver.getSolver().interpolateVelocity(m_M.col(k)));
	}

	modes.save("deim_modes.txt", arma::raw_ascii);

	for (arma::uword ind : m_indsP) {
		m_gridIndsP.push_back(rom_solver.getSolver().vectorToGridIndex(ind));
	}

	m_PsiTM = rom_solver.Psi().t() * m_M;

	m_M2 = m_PsiTM * m_M1;

	m_M4 = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * m_PsiTM.t()));

	m_M3 = m_PsiTM * m_M4;
}

arma::Mat<double> LSDEIM::Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Mat<double> PTJPsi;

	for (int m = 0; m < m_numModes; ++m) {
		PTJPsi = arma::join_cols(PTJPsi, rom_solver.Jindex(a, m_indsP[m], m_gridIndsP[m].first, m_gridIndsP[m].second));
	}

	arma::Col<double> PTN(m_numModes);

	for (int i = 0; i < m_numModes; ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);
	}

	double alfa = arma::as_scalar(a.t() * m_M3 * a);
	double beta = arma::as_scalar(a.t() * m_M2 * PTN);
	double gamma = beta / alfa;

	//std::cout << "1" << std::endl;

	arma::Mat<double> output = m_M1 * PTJPsi;

	//std::cout << "2" << std::endl;

	output -= gamma * m_M4;

	//std::cout << "3" << std::endl;

	arma::Col<double> v1 = (1.0 / alfa) * (m_M2 * PTN + (a.t() * m_M2 * PTJPsi).t()) - gamma / alfa * (m_M3 * a + (a.t() * m_M3).t());

	//std::cout << "4" << std::endl;

	output -= m_M4 * a * v1.t();

	//std::cout << "5" << std::endl;

	return m_PsiTM * output;
}