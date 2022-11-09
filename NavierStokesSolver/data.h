#ifndef H_DATA
#define H_DATA

#include <iostream>
#include <armadillo>
#include <vector>

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

	//load data from files...
	//create new dataCollector with a part of the data...

	dataCollector<COLLECT_DATA_FLAG> split(int, int) const;
	void loadDataMatrix(const std::string&);
	void loadOperatorMatrix(const std::string&);

	void appendDataLeft(const arma::Mat<double>&);
	void appendDataRight(const arma::Mat<double>&);
	void appendOperatorLeft(const arma::Mat<double>&);
	void appendOperatorRight(const arma::Mat<double>&);

	void clearData();
	void clearOperatorData();

};


template<bool COLLECT_DATA>
dataCollector<COLLECT_DATA> dataCollector<COLLECT_DATA>::split(int index1, int index2) const {

	dataCollector<COLLECT_DATA> res;

	res.m_dataMatrix = m_dataMatrix.cols(index1, index2);
	res.m_operatorMatrix = m_operatorMatrix.cols(index1, index2);

	//move?
	return res;
}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::clearData() {
	m_dataMatrix.clear();
}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::clearOperatorData() {
	m_operatorMatrix.clear();
}


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

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::loadDataMatrix(const std::string& name) {

	m_dataMatrix.load(name, arma::arma_binary);

}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::loadOperatorMatrix(const std::string& name) {

	m_operatorMatrix.load(name, arma::arma_binary);

}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::appendDataLeft(const arma::Mat<double>& data) {
	m_dataMatrix.insert_cols(0, data);
}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::appendDataRight(const arma::Mat<double>& data) {
	m_dataMatrix.insert_cols(m_dataMatrix.n_cols, data);
}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::appendOperatorLeft(const arma::Mat<double>& data) {
	m_operatorMatrix.insert_cols(0, data);
}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::appendOperatorRight(const arma::Mat<double>& data) {
	m_operatorMatrix.insert_cols(m_operatorMatrix.n_cols, data);
}


#endif
