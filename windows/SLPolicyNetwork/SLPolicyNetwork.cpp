#include <stdio.h>
#include <algorithm>
#include <random>
#include <vector>
#include <memory>
#include <intrin.h>
#include <cuda_runtime.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

using namespace caffe;
using namespace std;


DEFINE_int32(max_iter, 100, "max iterations");

const int batch_size = 16;
const int features_num = 4;

typedef short XY;
typedef unsigned long long Bitboard[6];

#pragma pack(push, 1)
struct Position
{
	XY xy;
	Bitboard player_color;
	Bitboard opponent_color;
};
#pragma pack(pop)

// 入力データ
float input_data[batch_size * 100][features_num][19][19];
float labels[batch_size * 100];
// テストデータ
float test_input_data[batch_size * 10][features_num][19][19];
float test_labels[batch_size * 10];

// 局面をシャッフルして読み込み
void load_input_features(const char* filename, Position* &position, int &size)
{
	FILE *fp = fopen(filename, "rb");
	if (fp == nullptr)
	{
		fprintf(stderr, "%s read error.\n", filename);
		exit(1);
	}

	fseek(fp, 0, SEEK_END);
	int filesize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	size = filesize / sizeof(Position);

	position = new Position[size];

	vector<int> index;
	for (int i = 0; i < size; i++)
	{
		index.push_back(i);
	}
	shuffle(index.begin(), index.end(), mt19937());

	for (int i = 0; i < size; i++)
	{
		fread(position + index[i], sizeof(Position), 1, fp);
	}

	fclose(fp);
}

void bitboard_to_array(Bitboard bitboard, float array[19][19])
{
	for (int i = 0; i < 19 * 19; i++)
	{
		int idx = i / 64;
		int p = i % 64;
		*((float*)array + i) = _bittest64((long long*)bitboard + idx, p);
	}
}

void prepare_input_data(Position* position, float input_data[features_num][19][19], float *label)
{
	bitboard_to_array(position->player_color, input_data[0]);
	bitboard_to_array(position->opponent_color, input_data[1]);

	// empty
	Bitboard empty;
	for (int i = 0; i < 6; i++)
	{
		empty[i] = ~(position->player_color[i] | position->opponent_color[i]);
	}
	bitboard_to_array(empty, input_data[2]);

	*label = position->xy;
}

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	GlobalInit(&argc, &argv);

	if (argc < 2)
	{
		return 1;
	}

	// 局面データ読み込み
	Position *position_train;
	int train_size;
	Position *position_test;
	int test_size;

	load_input_features(argv[1], position_train, train_size);
	load_input_features(argv[2], position_test, test_size);

	LOG(INFO) << "train data size = " << train_size;
	LOG(INFO) << "test data size = " << test_size;

	Caffe::set_mode(Caffe::GPU);

	// Solverの設定をテキストファイルから読み込む
	SolverParameter solver_param;
	ReadProtoFromTextFileOrDie("solver.prototxt", &solver_param);

	solver_param.set_max_iter(FLAGS_max_iter);

	std::shared_ptr<Solver<float>>
		solver(SolverRegistry<float>::CreateSolver(solver_param));
	const auto net = solver->net();
	const auto input_layer =
		boost::dynamic_pointer_cast<MemoryDataLayer<float>>(
		net->layer_by_name("input"));

	const auto test_net = solver->test_nets()[0];
	const auto test_input_layer =
		boost::dynamic_pointer_cast<MemoryDataLayer<float>>(
		test_net->layer_by_name("input"));

	// 学習
	int index = 0;
	int test_index = 0;
	for (int iter100 = 0; iter100 < FLAGS_max_iter / 100; iter100++)
	{
		// 入力データ準備
		for (int i = 0; i < batch_size * 100; i++, index++)
		{
			prepare_input_data(position_train + index, input_data[i], labels + i);
		}
		input_layer->Reset((float*)input_data, (float*)labels, batch_size * 100);

		// テストデータ準備
		for (int i = 0; i < batch_size * 10; i++, test_index++)
		{
			prepare_input_data(position_test + test_index, test_input_data[i], test_labels + i);
		}
		test_input_layer->Reset((float*)test_input_data, (float*)test_labels, batch_size * 10);

		if (iter100 == FLAGS_max_iter / 100 - 1)
		{
			solver->Solve();
		}
		else
		{
			solver->Step(100);
		}
	}

	return 0;
}