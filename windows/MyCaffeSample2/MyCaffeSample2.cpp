#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

using namespace caffe;
using namespace std;

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	GlobalInit(&argc, &argv);

	Caffe::set_mode(Caffe::CPU);

	// Solverの設定をテキストファイルから読み込む
	SolverParameter solver_param;
	ReadProtoFromTextFileOrDie("solver.prototxt", &solver_param);
	std::shared_ptr<Solver<float>>
		solver(SolverRegistry<float>::CreateSolver(solver_param));
	const auto net = solver->net();

	// 入力データ
	const int data_size = 100;
	const int input_num = 2;
	float input_data[data_size][input_num] = {
		{ 0.18, 0.15 }, { 0.20, 0.90 }, { 0.94, 0.11 }, { 0.96, 0.63 }, { 0.25, 0.72 }, { 0.83, 0.29 }, { 0.52, 0.30 }, { 0.38, 0.51 }, { 0.50, 0.18 }, { 0.38, 0.85 }, { 0.64, 0.15 }, { 0.14, 0.11 }, { 0.52, 0.42 }, { 0.69, 0.50 }, { 0.16, 0.45 }, { 0.52, 0.78 }, { 0.55, 0.15 }, { 0.05, 0.18 }, { 0.52, 0.72 }, { 0.07, 0.03 }, { 0.70, 0.73 }, { 0.15, 0.36 }, { 0.74, 0.73 }, { 0.07, 0.36 }, { 0.74, 0.13 }, { 0.33, 0.27 }, { 0.40, 0.53 }, { 0.65, 0.51 }, { 0.80, 0.12 }, { 0.75, 0.35 }, { 0.35, 0.28 }, { 0.71, 0.24 }, { 0.60, 0.89 }, { 0.08, 0.14 }, { 0.66, 0.47 }, { 0.43, 0.63 }, { 0.82, 0.08 }, { 0.88, 0.69 }, { 0.45, 0.63 }, { 0.76, 0.05 }, { 0.06, 0.76 }, { 0.27, 0.48 }, { 0.70, 0.34 }, { 0.02, 0.90 }, { 0.46, 0.50 }, { 0.11, 0.71 }, { 0.90, 0.17 }, { 0.41, 0.09 }, { 0.39, 0.59 }, { 0.03, 0.94 }, { 0.31, 0.36 }, { 0.88, 0.56 }, { 0.23, 0.91 }, { 0.02, 0.20 }, { 0.76, 0.14 }, { 0.96, 0.48 }, { 0.22, 0.69 }, { 0.59, 0.57 }, { 0.45, 0.37 }, { 0.84, 0.15 }, { 0.06, 0.85 }, { 0.20, 0.90 }, { 0.48, 0.80 }, { 0.22, 0.16 }, { 0.43, 0.19 }, { 0.83, 0.45 }, { 0.39, 0.02 }, { 0.81, 0.35 }, { 0.18, 0.01 }, { 0.52, 0.35 }, { 0.59, 0.34 }, { 0.90, 0.81 }, { 0.38, 0.67 }, { 0.93, 0.63 }, { 0.61, 0.48 }, { 0.34, 0.67 }, { 0.05, 0.73 }, { 0.36, 0.87 }, { 0.47, 0.58 }, { 0.76, 0.82 }, { 0.07, 0.15 }, { 0.37, 0.63 }, { 0.42, 0.99 }, { 0.00, 0.06 }, { 0.06, 0.08 }, { 0.68, 0.23 }, { 0.14, 0.52 }, { 0.02, 0.51 }, { 0.97, 0.42 }, { 0.87, 0.49 }, { 0.55, 0.51 }, { 0.10, 0.55 }, { 0.29, 0.92 }, { 0.38, 0.63 }, { 0.71, 0.56 }, { 0.36, 0.95 }, { 0.53, 0.83 }, { 0.18, 0.69 }, { 0.47, 0.70 }, { 0.13, 0.34 } };
	float label[data_size] = { 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1 };
	const auto input_layer =
		boost::dynamic_pointer_cast<MemoryDataLayer<float>>(
		net->layer_by_name("input"));
	input_layer->Reset((float*)input_data, (float*)label, data_size);

	// 学習
	LOG(INFO) << "Solve start.";
	solver->Solve();

	// パラメータ表示
	const auto hidden1_blobs = net->layer_by_name("hidden1")->blobs();
	const auto hidden1_weight = hidden1_blobs[0]->cpu_data();
	const auto hidden1_bias = hidden1_blobs[1]->cpu_data();
	LOG(INFO) << "hidden1_weight shape = ";
	for (auto v : hidden1_blobs[0]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden1_weight = "
		<< hidden1_weight[0] << ", " << hidden1_weight[1] << ", "
		<< hidden1_weight[2] << ", " << hidden1_weight[3] << ", "
		<< hidden1_weight[4] << ", " << hidden1_weight[5];
	LOG(INFO) << "hidden1_bias shape = ";
	for (auto v : hidden1_blobs[1]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden1_bias = " << hidden1_bias[0] << ", " << hidden1_bias[1] << ", " << hidden1_bias[2];

	const auto hidden2_blobs = net->layer_by_name("hidden2")->blobs();
	const auto hidden2_weight = hidden2_blobs[0]->cpu_data();
	const auto hidden2_bias = hidden2_blobs[1]->cpu_data();
	LOG(INFO) << "hidden2_weight shape = ";
	for (auto v : hidden2_blobs[0]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden2_weight = "
		<< hidden2_weight[0] << ", " << hidden2_weight[1] << ", "
		<< hidden2_weight[2] << ", " << hidden2_weight[3] << ", "
		<< hidden2_weight[4] << ", " << hidden2_weight[5];
	LOG(INFO) << "hidden2_bias shape = ";
	for (auto v : hidden2_blobs[1]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden2_bias = " << hidden2_bias[0] << ", " << hidden2_bias[1];


	// 予測
	Net<float> net_test("net.prototxt", TEST);
	net_test.CopyTrainedLayersFrom("_iter_1000.caffemodel");

	const auto input_test_layer =
		boost::dynamic_pointer_cast<MemoryDataLayer<float>>(
		net_test.layer_by_name("input"));
	for (int batch = 0; batch < 10; batch++)
	{
		input_test_layer->Reset((float*)input_data + batch * 20, (float*)label + batch * 10, 10);

		const auto result = net_test.Forward();

		const auto data = result[1]->cpu_data();
		for (int i = 0; i < 10; i++)
		{
			LOG(INFO) << data[i * 2] << ", " << data[i * 2 + 1] << ", " << (data[i * 2] < data[i * 2 + 1]) ? 0 : 1;
		}
	}

	return 0;
}