#ifndef H_DATA
#define H_DATA

#include <iostream>
#include <armadillo>

template<bool COLLECT_DATA_FLAG>
class dataCollector {
private:
	arma::Mat<double> m_dataMatrix;

public:
	static constexpr bool COLLECT_DATA = COLLECT_DATA_FLAG;

	void addColumn(const arma::Col<double>&);
	const arma::Mat<double>& getDataMatrix() const;

};

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::addColumn(const arma::Col<double>& vel) {

	m_dataMatrix.insert_cols(m_dataMatrix.n_cols, vel);

}

template<bool COLLECT_DATA>
const arma::Mat<double>& dataCollector<COLLECT_DATA>::getDataMatrix() const {

	return m_dataMatrix;

}


#endif
