#ifndef H_DATA
#define H_DATA

#include <iostream>
#include <armadillo>
#include <torch/torch.h>
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

	void clearData();
	void clearOperatorData();

	torch::Tensor toTensor(const arma::Mat<double>&, const torch::Device&) const;

};

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::clearData() {
	m_dataMatrix.clear();
}

template<bool COLLECT_DATA>
void dataCollector<COLLECT_DATA>::clearOperatorData() {
	m_operatorMatrix.clear();
}

template<bool COLLECT_DATA> 
torch::Tensor dataCollector<COLLECT_DATA>::toTensor(const arma::Mat<double>& data, const torch::Device& device) const {

	torch::Tensor out;

	if (data.n_cols != 1) {
		std::vector<int64_t> dims = { static_cast<int64_t>(data.n_rows), static_cast<int64_t>(data.n_cols) };
		out = torch::zeros(dims, device);

		for (int i = 0; i < data.n_rows; ++i) {
			for (int j = 0; j < data.n_cols; ++j) {
				out.index_put_({ static_cast<int64_t>(i), static_cast<int64_t>(j) }, data(i, j));
			}
		}
	}
	else {
		std::vector<int64_t> dims = { static_cast<int64_t>(data.n_rows) };
		out = torch::zeros(dims, device);

		for (int i = 0; i < data.n_rows; ++i) {
			out.index_put_({ static_cast<int64_t>(i) }, data(i, 0));
		}
	}


	return out;
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


#endif
