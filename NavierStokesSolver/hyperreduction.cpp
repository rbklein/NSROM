#include <iostream>
#include <armadillo>
#include <algorithm>

#include "mesh.h"
#include "solver.h"
#include "ROM.h"

//#define LOAD_DATA

//#define L2_REGULARIZATION

HYPER_REDUCTION_METHOD Base_hyperReduction::getType() const {
	return m_method;
}

arma::Col<double> noHyperReduction::Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {
	return rom_solver.Psi().t() * rom_solver.getSolver().N(rom_solver.Psi() * a);
}

arma::Col<double> noHyperReduction::N(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {
	return rom_solver.getSolver().N(rom_solver.Psi() * a);
}

arma::Mat<double> noHyperReduction::Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {
	return arma::Mat<double>(rom_solver.Psi().t() * rom_solver.getSolver().J(rom_solver.Psi() * a) * rom_solver.Psi()) ;
}

void noHyperReduction::initialize(const ROM_Solver& rom_solver) {
	//no initialization steps necessary
}

int generateNumber(const std::vector<arma::uword>& vec, int n) {
	//std::random_device rd;
	//std::default_random_engine generator(rd());
	std::default_random_engine generator;
	std::uniform_int_distribution<> distribution(0, n - 1);
	bool found = false;
	int number;
	while (!found) {
		number = distribution(generator);
		found = true;
		for (auto it : vec) {
			if (it == number) {
				found = false;
				break;
			}
		}
	}
	return number;
}

void DEIM::setupMeasurementSpace() {

#ifndef LOAD_DATA

	//get reference to data matrix
	const arma::Mat<double>& operatorMatrix = m_collector.getOperatorMatrix();

	//arma::Mat<double> operatorMatrix;
	//operatorMatrix.load("operator_snapshots_slr");  //.load("operator_snapshots_slr_Re100");  //.load("operator_snapshots_2dturb_" + std::to_string(m_datasetIndex)); //

	//declare singular vectors and singular values
	arma::Mat<double> MFull, _;
	arma::Col<double> singularValues;

	//perform svd of snapshots
	arma::svd_econ(MFull, singularValues, _, operatorMatrix, "left", "std");

	if (m_saveM)
		MFull.save("MFull.bin", arma::arma_binary);

	//singularValues.save("deim_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::accu(singularValues.rows(0, m_numModes - 1)) / arma::accu(singularValues);

	std::cout << "DEIM RIC: " << m_RIC * 100.0 << "% " << std::endl;

#else

	arma::Mat<double> MFull;
	MFull.load("MFull.bin", arma::arma_binary);

	arma::Col<double> singularValues;
	singularValues.load("deim_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::accu(singularValues.rows(0, m_numModes - 1)) / arma::accu(singularValues);

	std::cout << "DEIM RIC: " << m_RIC * 100.0 << "% " << std::endl;

#endif

	//get amount of unknowns
	arma::uword n = MFull.n_rows;

	//set size of measurement space matrix
	m_P = arma::SpMat<double>(n, m_numModes);

	//find max index of first data matrix column
	arma::uword ind = arma::abs(MFull.col(0)).index_max();

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

	//std::cout << "diff operators: " << arma::norm(m_M * c - rom_solver.getSolver().N(rom_solver.Psi() * a), "inf") << std::endl;

	return m_PsiTM * c;
}

arma::Col<double> DEIM::N(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Col<double> c(m_numModes);

	arma::Col<double> PTN(m_numModes);

	for (int i = 0; i < m_numModes; ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);
	}

	c = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * PTN));

	return m_M * c;
}

void DEIM::initialize(const ROM_Solver& rom_solver) {

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

#ifndef LOAD_DATA

	//get reference to data matrix
	const arma::Mat<double>& operatorMatrix = m_collector.getOperatorMatrix();

	//arma::Mat<double> operatorMatrix;
	//operatorMatrix.load("operator_snapshots_slr_Re100"); //.load("operator_snapshots_2dturb_" + std::to_string(m_datasetIndex));  //

	//declare singular vectors and singular values
	arma::Mat<double> MFull, _;
	arma::Col<double> singularValues;

	//perform svd of snapshots
	arma::svd_econ(MFull, singularValues, _, operatorMatrix, "left", "std");

	if (m_saveM)
		MFull.save("MFull.bin", arma::arma_binary);

	//singularValues.save("deim_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::accu(singularValues.rows(0, m_numModes - 1)) / arma::accu(singularValues);

	std::cout << "SMDEIM RIC: " << m_RIC * 100.0 << "% " << std::endl;

#else

	arma::Mat<double> MFull;
	MFull.load("MFull.bin", arma::arma_binary);

	arma::Col<double> singularValues;
	singularValues.load("deim_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::accu(singularValues.rows(0, m_numModes - 1)) / arma::accu(singularValues);

	std::cout << "SMDEIM RIC: " << m_RIC * 100.0 << "% " << std::endl;

#endif

	//get amount of unknowns
	arma::uword n = MFull.n_rows;

	//set size of measurement space matrix
	m_P = arma::SpMat<double>(n, m_numModes - 1);

	//find max index of first data matrix column
	arma::uword ind = arma::abs(MFull.col(0)).index_max();

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

	std::cout << aTPsiTM(m_numModes - 1) << std::endl;

	//std::cout << ra.t() << std::endl;

	MpiPTN = arma::solve(arma::trimatu(m_Mp_U), arma::solve(arma::trimatl(m_Mp_L), m_Mp_perm * PTN));

	c.rows(0, m_numModes - 2) = MpiPTN - arma::as_scalar(ra.t() * MpiPTN) / (1.0 + arma::as_scalar(ra.t() * m_MpiMm)) * m_MpiMm;

	c.row(m_numModes - 1) = arma::as_scalar(ra.t() * c.rows(0, m_numModes - 2));

	//std::cout << "diff operators: " << arma::norm(m_M * c - rom_solver.getSolver().N(rom_solver.Psi() * a), "inf") << std::endl;

	return m_PsiTM * c;
}

arma::Col<double> SPDEIM::N(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Col<double> c(m_numModes);

	arma::Col<double> MpiPTN;

	arma::Col<double> ra(m_numModes - 1);

	arma::Col<double> PTN(m_numModes - 1);

	arma::Row<double> aTPsiTM = a.t() * m_PsiTM;

	for (int i = 0; i < (m_numModes - 1); ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);

		ra(i) = -aTPsiTM(i) / aTPsiTM(m_numModes - 1);
	}

	MpiPTN = arma::solve(arma::trimatu(m_Mp_U), arma::solve(arma::trimatl(m_Mp_L), m_Mp_perm * PTN));

	c.rows(0, m_numModes - 2) = MpiPTN - arma::as_scalar(ra.t() * MpiPTN) / (1.0 + arma::as_scalar(ra.t() * m_MpiMm)) * m_MpiMm;

	c.row(m_numModes - 1) = arma::as_scalar(ra.t() * c.rows(0, m_numModes - 2));

	return m_M * c;
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

	for (arma::uword ind : m_indsP) {
		m_gridIndsP.push_back(rom_solver.getSolver().vectorToGridIndex(ind));
	}

	m_PsiTM = rom_solver.Psi().t() * m_M;
}





void LSDEIM::setupMeasurementSpace() {

#ifndef LOAD_DATA

	//get reference to data matrix
	const arma::Mat<double>& operatorMatrix = m_collector.getOperatorMatrix();

	//arma::Mat<double> operatorMatrix;
	//operatorMatrix.load("operator_snapshots_2dturb_" + std::to_string(m_datasetIndex));  //.load("operator_snapshots_slr_Re100"); //

	//declare singular vectors and singular values
	arma::Mat<double> MFull, _;
	arma::Col<double> singularValues;

	//perform svd of snapshots
	arma::svd_econ(MFull, singularValues, _, operatorMatrix, "left", "std");

	if (m_saveM)
		MFull.save("MFull.bin", arma::arma_binary);

	singularValues.save("deim_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::accu(singularValues.rows(0, m_numModes - 1)) / arma::accu(singularValues);

	std::cout << "LSDEIM RIC: " << m_RIC * 100.0 << "% " << std::endl;

#else

	arma::Mat<double> MFull;
	MFull.load("MFull.bin", arma::arma_binary);

	arma::Col<double> singularValues;
	singularValues.load("deim_sing_vals.txt", arma::raw_ascii);

	m_RIC = arma::accu(singularValues.rows(0, m_numModes - 1)) / arma::accu(singularValues);

	std::cout << "LSDEIM RIC: " << m_RIC * 100.0 << "% " << std::endl;

#endif

	//get amount of unknowns
	arma::uword n = MFull.n_rows;

	//set size of measurement space matrix
	m_P = arma::SpMat<double>(n, m_numPoints);

	//find max index of first data matrix column
	arma::uword ind = arma::abs(MFull.col(0)).index_max();

	m_indsP.push_back(ind);

	//update measurement space
	m_P(ind, 0) = 1.0;

	//declare c vector
	arma::Col<double> c;

	//int index_p = (m_numPoints > MFull.n_cols) ? MFull.n_cols : m_numPoints;

	int index_p = (m_numPoints > m_numModes) ? m_numModes : m_numPoints;

	for (arma::uword i = 1; i < index_p; ++i) {
		//equate next column and previous modes in measurement space (not sure why this does not solve for m_numModes components, but it seems to work)
		c = arma::solve(m_P.t() * MFull.cols(0, i - 1), (m_P.t() * MFull.col(i)).as_col());

		//find index of max absolute difference in FOM space
		ind = arma::abs(MFull.col(i) - MFull.cols(0, i - 1) * c).index_max();

		m_indsP.push_back(ind);

		//update measurement space
		m_P(ind, i) = 1.0;
	}

	for (int i = index_p; i < m_numPoints; ++i) {

		ind = generateNumber(m_indsP, n); //only 256 x 256 x 2 - 1??

		//std::cout << ind << std::endl;

		m_indsP.push_back(ind);

		m_P(ind, i) = 1.0;
	}

	//define deim modes
	m_M = MFull.cols(0, m_numModes - 1);

	//lu decompose deim modes in measurement space for fast inversion (ADDED weighted L2 REGULARISATION TERM)
	arma::lu(m_PTM_L, m_PTM_U, m_PTM_perm, 2.0 * ((m_P.t() * m_M).t() * (m_P.t() * m_M) 

#ifndef L2_REGULARIZATION 

		)); 

	#else 
		
		+1e-6 * arma::diagmat(arma::pow(arma::linspace(1, m_numModes, m_numModes), 2.0))));// arma::eye(m_numModes, m_numModes))); arma::linspace(1, m_numModes, m_numModes); arma::min(singularValues(0) / singularValues.rows(0,m_numModes - 1), 

#endif

	m_PTM = m_P.t() * m_M;

	m_M1 = 2.0 * arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * m_PTM.t()));

}

arma::Col<double> LSDEIM::Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Col<double> c(m_numModes);

	arma::Col<double> PTN(m_indsP.size());

	for (int i = 0; i < m_indsP.size(); ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);
	}

	arma::Col<double> PTMPTN = 2.0 * m_PTM.t() * PTN;

	arma::Col<double> B = (a.t() * m_PsiTM).t();

	arma::Col<double> AiPTMPTN = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * PTMPTN));

	arma::Col<double> AiB = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * B));

	c = AiPTMPTN - arma::as_scalar((B.t() * AiPTMPTN) / (B.t() * AiB)) * AiB;

	return m_PsiTM * c;
}


//correct back to 'return m_M * c;'
arma::Col<double> LSDEIM::N(const arma::Col<double>& a, const ROM_Solver& rom_solver) const {

	arma::Col<double> c(m_numModes);

	arma::Col<double> PTN(m_indsP.size());

	for (int i = 0; i < m_indsP.size(); ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);
	}

	arma::Col<double> PTMPTN = 2.0 * m_PTM.t() * PTN;

	arma::Col<double> B = (a.t() * m_PsiTM).t();

	arma::Col<double> AiPTMPTN = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * PTMPTN));

	arma::Col<double> AiB = arma::solve(arma::trimatu(m_PTM_U), arma::solve(arma::trimatl(m_PTM_L), m_PTM_perm * B));

	c = AiPTMPTN - arma::as_scalar((B.t() * AiPTMPTN) / (B.t() * AiB)) * AiB;

	return m_M * c;
}

void LSDEIM::initialize(const ROM_Solver& rom_solver) {

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

	for (int m = 0; m < m_indsP.size(); ++m) {
		PTJPsi = arma::join_cols(PTJPsi, rom_solver.Jindex(a, m_indsP[m], m_gridIndsP[m].first, m_gridIndsP[m].second));
	}

	arma::Col<double> PTN(m_indsP.size());

	for (int i = 0; i < m_indsP.size(); ++i) {
		PTN(i) = rom_solver.Nindex(a, m_indsP[i], m_gridIndsP[i].first, m_gridIndsP[i].second);
	}

	double alfa = arma::as_scalar(a.t() * m_M3 * a);
	double beta = arma::as_scalar(a.t() * m_M2 * PTN);
	double gamma = beta / alfa;

	arma::Mat<double> output = m_M1 * PTJPsi;

	output -= gamma * m_M4;

	arma::Col<double> v1 = (1.0 / alfa) * (m_M2 * PTN + (a.t() * m_M2 * PTJPsi).t()) - gamma / alfa * (m_M3 * a + (a.t() * m_M3).t());

	output -= m_M4 * a * v1.t();

	return m_PsiTM * output;
}

