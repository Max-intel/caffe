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
	const int data_size = 8;
	const int input_num = 2;
	float input_data[data_size][input_num] = {
		{ 0.8, 0.1 },
		{ 0.2, 0.9 },
		{ 0.7, 0.2 },
		{ 0.1, 0.8 },
		{ 0.3, 0.7 },
		{ 0.9, 0.1 },
		{ 0.3, 0.8 },
		{ 0.8, 0.2 }
	};
	float label[data_size] = {	0, 1, 0, 1, 1, 0, 1, 0 };
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
	LOG(INFO) << "hidden1_weight = " << hidden1_weight[0] << ", " << hidden1_weight[1];
	LOG(INFO) << "hidden1_bias shape = ";
	for (auto v : hidden1_blobs[1]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden1_bias = " << hidden1_bias[0];

	const auto hidden2_blobs = net->layer_by_name("hidden2")->blobs();
	const auto hidden2_weight = hidden2_blobs[0]->cpu_data();
	const auto hidden2_bias = hidden2_blobs[1]->cpu_data();
	for (auto v : hidden2_blobs[0]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden2_weight = " << hidden2_weight[0] << ", " << hidden2_weight[1];
	for (auto v : hidden2_blobs[1]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden2_bias = " << hidden2_bias[0] << ", " << hidden2_bias[1];

	return 0;
}