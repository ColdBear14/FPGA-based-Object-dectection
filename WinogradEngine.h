#ifndef WINOGRADENGINE_H
#define WINOGRADENGINE_H

#include "global.h"

void input_transform(Tile4x4 d, Tile4x4 &U);

// void weight_transform(Tile3x3 g, Tile4x4 &V);

void ewmm(Tile4x4 U, Tile4x4 V, Tile4x4 &M);

void output_transform(Tile4x4 M, Tile2x2 &Y);

void winograd_engine_top(
    hls::stream<Tile4x4> &in_tile_stream,  // Luồng dữ liệu ảnh đầu vào
    hls::stream<Tile4x4> &weight_v_stream, // Trọng số
    hls::stream<Tile2x2> &out_tile_stream,  // Kết quả ra 2x2
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<int> &num_tiles_stream,
    hls::stream<int> &cin_stream,     // MỚI: Thêm stream cấu hình Cin
    hls::stream<int> &cout_stream     // MỚI: Thêm stream
);


#endif 
