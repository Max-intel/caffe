#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/bias_layer.hpp>

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
	const int data_size = 10;
	const int input_num = 2;
	float input_data[data_size][input_num] = {
		{ 0.18, 0.15 }, { 0.20, 0.90 }, { 0.94, 0.11 }, { 0.96, 0.63 }, { 0.25, 0.72 }, { 0.83, 0.29 }, { 0.52, 0.30 }, { 0.38, 0.51 }, { 0.50, 0.18 }, { 0.38, 0.85 } };
	float label[data_size] = { 1, 0, 0, 0, 0, 0, 1, 1, 1, 0 };
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
	LOG(INFO) << "hidden2_weight shape = ";
	for (auto v : hidden2_blobs[0]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "hidden2_weight = "
		<< hidden2_weight[0] << ", " << hidden2_weight[1] << ", "
		<< hidden2_weight[2] << ", " << hidden2_weight[3] << ", "
		<< hidden2_weight[4] << ", " << hidden2_weight[5];

	const auto bias_blobs = net->layer_by_name("bias")->blobs();
	const auto bias_bias = bias_blobs[0]->cpu_data();
	LOG(INFO) << "bias_bias shape = ";
	for (auto v : bias_blobs[0]->shape())
		LOG(INFO) << v;
	LOG(INFO) << "bias_bias = " << bias_bias[0] << ", " << bias_bias[1] << ", " << bias_bias[2];


	// 予測
	Net<float> net_test("net.prototxt", TEST);
	net_test.CopyTrainedLayersFrom("_iter_10.caffemodel");

	const auto input_test_layer =
		boost::dynamic_pointer_cast<MemoryDataLayer<float>>(
		net_test.layer_by_name("input"));

	input_test_layer->Reset((float*)input_data, (float*)label, 10);

	const auto result = net_test.Forward();

	const auto data = result[1]->cpu_data();
	for (int i = 0; i < 10; i++)
	{
		LOG(INFO) << data[i * 2] << ", " << data[i * 2 + 1] << ", " << (data[i * 2] < data[i * 2 + 1]) ? 0 : 1;
	}

	return 0;
}