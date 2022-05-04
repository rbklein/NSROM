#ifndef H_ROM
#define H_ROM

#include <iostream>
#include <armadillo>

#include "solver.h"
#include "data.h"

enum class HYPER_REDUCTION_METHOD {
	NONE,
	EXACT_TENSOR_DECOMPOSITION,
	DEIM,
	SPDEIM,
	LSDEIM,
};

//forward declaration of ROM_Solver class
class ROM_Solver;


//base class for all hyper-reduction methods
class Base_hyperReduction
{
protected:

	HYPER_REDUCTION_METHOD m_method;

	Base_hyperReduction(HYPER_REDUCTION_METHOD method)
		: m_method(method)
	{}

public:

	virtual arma::Col<double> Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const = 0;

	virtual arma::Mat<double> Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const = 0;

	virtual void initialize(const ROM_Solver& rom_solver) = 0;

	HYPER_REDUCTION_METHOD getType() const;
};




//Naive ROM implementation
class noHyperReduction : public Base_hyperReduction
{
public:

	noHyperReduction()
		: Base_hyperReduction(HYPER_REDUCTION_METHOD::NONE)
	{}

	virtual arma::Col<double> Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

	virtual arma::Mat<double> Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

	virtual void initialize(const ROM_Solver& rom_solver) override;

};





//Discrete Empirical Interpolation Method
class DEIM : public Base_hyperReduction
{
protected:
	//number of modes
	int m_numModes;

	//snapshot data collector
	const dataCollector<true>& m_collector;

	//measurement space and vector indices
	arma::SpMat<double> m_P;
	std::vector<arma::uword> m_indsP;
	std::vector<std::pair<arma::uword, arma::uword>> m_gridIndsP;

	//DEIM modes
	arma::Mat<double> m_M;

	//projected DEIM modes
	arma::Mat<double> m_PsiTM;

	//DEIM modes in measurement space
	arma::Mat<double> m_PTM_L;
	arma::Mat<double> m_PTM_U;
	arma::Mat<double> m_PTM_perm;

public:

	DEIM(int numModes, const dataCollector<true>& collector)
		: Base_hyperReduction(HYPER_REDUCTION_METHOD::DEIM),
		m_numModes(numModes),
		m_collector(collector)
	{
		setupMeasurementSpace();
	}

	virtual arma::Col<double> Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

	virtual void initialize(const ROM_Solver& rom_solver) override;

	virtual arma::Mat<double> Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

private:

	void setupMeasurementSpace();

};





//Structure-Preserving Discrete Empirical Interpolation Method
class SPDEIM : public Base_hyperReduction
{
protected:
	//number of modes
	int m_numModes;

	//snapshot data collector
	const dataCollector<true>& m_collector;

	//measurement space and vector indices
	arma::SpMat<double> m_P;
	std::vector<arma::uword> m_indsP;
	std::vector<std::pair<arma::uword, arma::uword>> m_gridIndsP;

	//SPDEIM modes
	arma::Mat<double> m_M;

	//POD projected DEIM modes
	arma::Mat<double> m_PsiTM;
	
	//Measurement space projected DEIM modes
	arma::Mat<double> m_PTM;

	//DEIM modes in measurement space
	arma::Mat<double> m_Mp_L;
	arma::Mat<double> m_Mp_U;
	arma::Mat<double> m_Mp_perm;

	//the m_numModes^th DEIM mode times inverse of Mp
	arma::Col<double> m_MpiMm;

public:

	SPDEIM(int numModes, const dataCollector<true>& collector)
		: Base_hyperReduction(HYPER_REDUCTION_METHOD::SPDEIM),
		m_numModes(numModes),
		m_collector(collector)
	{
		setupMeasurementSpace();
	}

	virtual arma::Col<double> Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

	virtual arma::Mat<double> Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

	virtual void initialize(const ROM_Solver& rom_solver) override;

private:

	void setupMeasurementSpace();

};





//Discrete Empirical Interpolation Method
class LSDEIM : public Base_hyperReduction
{
protected:
	//number of modes
	int m_numModes;

	//snapshot data collector
	const dataCollector<true>& m_collector;

	//measurement space and vector indices
	arma::SpMat<double> m_P;
	std::vector<arma::uword> m_indsP;
	std::vector<std::pair<arma::uword, arma::uword>> m_gridIndsP;

	//DEIM modes
	arma::Mat<double> m_M;

	//projected DEIM modes
	arma::Mat<double> m_PsiTM;
	arma::Mat<double> m_PTM;

	//DEIM modes in measurement space
	arma::Mat<double> m_PTM_L;
	arma::Mat<double> m_PTM_U;
	arma::Mat<double> m_PTM_perm;

	//jacobian precalculation matrices
	arma::Mat<double> m_M1;
	arma::Mat<double> m_M2;
	arma::Mat<double> m_M3;
	arma::Mat<double> m_M4;

public:

	LSDEIM(int numModes, const dataCollector<true>& collector)
		: Base_hyperReduction(HYPER_REDUCTION_METHOD::LSDEIM),
		m_numModes(numModes),
		m_collector(collector)
	{
		setupMeasurementSpace();
	}

	virtual arma::Col<double> Nrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

	virtual void initialize(const ROM_Solver& rom_solver) override;

	virtual arma::Mat<double> Jrh(const arma::Col<double>& a, const ROM_Solver& rom_solver) const override;

private:

	void setupMeasurementSpace();

};



//class that solves ROM calculations 
class ROM_Solver {

private:

	//reduced operators
	arma::Mat<double> m_Dr;

	//mode numbers
	int m_numModesPOD;

	//FOM solver reference
	const solver& m_solver;

	//truncated and structure-preserving POD basis
	arma::Mat<double> m_Psi;

	//data collector reference
	const dataCollector<true>& m_dataCollector;

	//hyper reduction algorithm
	Base_hyperReduction& m_hyperReduction;

public:

	ROM_Solver(const solver& solver, const dataCollector<true>& dataCollector, int numModesPOD, Base_hyperReduction& hyperReduction)
		:	m_solver(solver),
			m_dataCollector(dataCollector),
			m_numModesPOD(numModesPOD),
			m_hyperReduction(hyperReduction)
	{
		setupBasis();

		m_hyperReduction.initialize(*this);

		precomputeOperators();
	}
		
	arma::Col<double> calculateIC(const arma::Col<double>&) const;

	const arma::Mat<double>& Dr() const;
	const arma::Mat<double>& Psi() const;

	arma::Col<double> Nr(const arma::Col<double>&) const;
	arma::Mat<double> Jr(const arma::Col<double>&) const;

	double Nindex(const arma::Col<double>&, arma::uword, arma::uword, arma::uword) const;
	arma::Row<double> Jindex(const arma::Col<double>&, arma::uword, arma::uword, arma::uword) const;

	double nu() const;

	const solver& getSolver() const;

	double getRICs() const;

private:

	void setupBasis();

	void precomputeOperators();

};









#endif
