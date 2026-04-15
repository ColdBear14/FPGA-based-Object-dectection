#ifndef HELPER_H
#define HELPER_H

#include "global.h"

#include "DataRouter.h"
#include "WinogradEngine.h"
#include "LineBuffer.h"
#include "WeightRAM.h"
#include "SystolicEngine.h"
#include "Fuse.h"
#include "Scheduler.h"
#include "TopSystem.h"


using namespace std;

void golden_winograd(Tile4x4 d, Tile3x3 g, Tile2x2 &y_ref);
void weight_transform(Tile3x3 g, Tile4x4 &V);
void data_producer(hls::stream<Tile4x4> &out_stream, int num_tiles);
void weight_producer(hls::stream<weight_mat_t> &out_stream, int num_weight_vectors);

void fill_pixel_stream(hls::stream<pixel_t>& stream, int width, int height, int Cin);
void fill_weight_stream(hls::stream<weight_t>& stream, int kernel_size, int Cin, int Cout);
void fill_Wino_weight_stream(hls::stream<weight_t>& input, int kernel_size, int Cin, int Cout, hls::stream<weight_t>& output);

void read_output_stream(hls::stream<fuse_vec_out_t>& stream, int expected_count);

#endif