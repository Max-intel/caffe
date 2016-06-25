#include <iostream>
#include <memory>
#include <random>
#include <cuda_runtime.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

using namespace caffe;
using namespace std;

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	::google::ParseCommandLineFlags(&argc, &argv, true);

	int count;
	cudaGetDeviceCount(&count);

	cout << "cudaGetDeviceCount = " << count << endl;
	if (count == 0) {
		return 1;
	}

	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);
	cout << "GPU device name: " << device_prop.name << endl;

	Caffe::set_mode(Caffe::GPU);


	// ���t�f�[�^�Ƃ��ėp������̓f�[�^�ƖڕW�f�[�^��float�z��Ƃ��ď�������D
	// ���̓f�[�^�F2����
	// �ڕW�f�[�^�F1����
	const auto kMinibatchSize = 32;
	const auto kDataSize = kMinibatchSize * 10;
	std::array<float, kDataSize * 2> input_data;
	std::array<float, kDataSize> target_data;
	std::mt19937 random_engine;
	std::uniform_real_distribution<float> dist(0.0, 1.0);
	// 3x - 2y + 4 = target �ɏ]���ăf�[�^�𐶐�����D
	for (auto i = 0; i < kDataSize; ++i) {
		const float x = dist(random_engine);
		const float y = dist(random_engine);
		const float target = 3 * x - 2 * y + 4;
		input_data[i * 2] = x;
		input_data[i * 2 + 1] = y;
		target_data[i] = target;
	}
	// MemoryDataLayer�̓�������̒l���o�͂ł���DataLayer�D
	// �eMemoryDataLayer�ɂ͓��̓f�[�^�ƃ��x���f�[�^�i1�����̐����j��2��^����K�v�����邪�C
	// �����ł͉�A���s�������̂ŁC���̓f�[�^�ƖڕW�f�[�^���ꂼ���ʂ�MemoryDataLayer�ŏo�͂��C
	// ���x���f�[�^�̑���Ɏg�p����Ȃ��_�~�[�̒l��^���Ă����D
	std::array<float, kDataSize> dummy_data;
	std::fill(dummy_data.begin(), dummy_data.end(), 0.0f);

	// Solver�̐ݒ���e�L�X�g�t�@�C������ǂݍ���
	caffe::SolverParameter solver_param;
	caffe::ReadProtoFromTextFileOrDie("solver.prototxt", &solver_param);
	std::shared_ptr<caffe::Solver<float> >
		solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	const auto net = solver->net();
	// ���̓f�[�^��MemoryDataLayer"input"�ɃZ�b�g����
	const auto input_layer =
		boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
		net->layer_by_name("input"));
	assert(input_layer);
	input_layer->Reset(input_data.data(), dummy_data.data(), kDataSize);
	// �ڕW�f�[�^��MemoryDataLayer"target"�ɃZ�b�g����
	const auto target_layer =
		boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
		net->layer_by_name("target"));
	assert(target_layer);
	target_layer->Reset(target_data.data(), dummy_data.data(), kDataSize);

	// Solver�̐ݒ�ʂ�Ɋw�K���s��
	solver->Solve();

	// �w�K���ꂽ�p�����[�^���o�͂��Ă݂�
	// ax + by + c = target
	const auto ip_blobs = net->layer_by_name("ip")->blobs();
	const auto learned_a = ip_blobs[0]->cpu_data()[0];
	const auto learned_b = ip_blobs[0]->cpu_data()[1];
	const auto learned_c = ip_blobs[1]->cpu_data()[0];
	std::cout << learned_a << "x + " << learned_b << "y + " << learned_c
		<< " = target" << std::endl;

	// �w�K���ꂽ���f�����g���ė\�����Ă݂�
	// x = 10, y = 20
	std::array<float, kDataSize * 2> sample_input;
	sample_input[0] = 10;
	sample_input[1] = 20;
	input_layer->Reset(sample_input.data(), dummy_data.data(), kDataSize);
	net->ForwardPrefilled(nullptr);
	std::cout << "10a + 20b + c = " << net->blob_by_name("ip")->cpu_data()[0] << std::endl;

	return 0;
}