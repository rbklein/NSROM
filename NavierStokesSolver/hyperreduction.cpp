
/*
#include <iostream>
#include <armadillo>

#include "mesh.h"
#include "solver.h"
#include "ROM.h"

HYPER_REDUCTION_METHOD Base_HyperReduction::getMethod() const {
	return m_method;
}

arma::Col<double> Naive::Nr(const arma::Col<double>& a, const ROM_Solver& solver) const {
	return solver.Psi().t() * solver.getSolver().N(solver.Psi() * a);
}
*/