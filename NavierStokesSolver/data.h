#ifndef H_DATA
#define H_DATA

#include <iostream>
#include <armadillo>

//make a freeData() member function to save memory during run-time
template<bool COLLECT_DATA_FLAG>
class dataCollector {
private:
	arma::Mat<double> m_dataMatrix;
	arma::Mat<double> m_operatorMatrix;

public:
	static constexpr bool COLLECT_DATA = COLLECT_DATA_FLAG;

	void addColumn(const arma::Col<double>&);
	const arma::Mat<double>& getDataMatrix() const;

	void addOperatorColumn(const arma::Col<double>&);
	const arma::Mat<double>& getOperatorMatrix() const;

};

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::addColumn(const arma::Col<double>& vel) {

	m_dataMatrix.insert_cols(m_dataMatrix.n_cols, vel);

}

template<bool COLLECT_DATA>
const arma::Mat<double>& dataCollector<COLLECT_DATA>::getDataMatrix() const {

	return m_dataMatrix;

}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::addOperatorColumn(const arma::Col<double>& N) {

	m_operatorMatrix.insert_cols(m_operatorMatrix.n_cols, N);

}

template<bool COLLECT_DATA>
const arma::Mat<double>& dataCollector<COLLECT_DATA>::getOperatorMatrix() const {

	return m_operatorMatrix;

}


#endif
