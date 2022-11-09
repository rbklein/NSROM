#ifndef H_PLOT
#define H_PLOT

#include <iostream>
#include <armadillo>

#include "solver.h"

void plot(const solver&, const arma::Col<double>&, bool = false);

#endif
