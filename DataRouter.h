#ifndef DATAROUTER_H
#define DATAROUTER_H

#include "global.h"

void weight_demux(
    hls::stream<weight_mat_t> &in_weight_stream, // Từ WeightRAM
    hls::stream<ap_uint<2>>& mode_stream,                    // 0: Systolic, 1: Winograd
    hls::stream<weight_mat_t> &out_systolic_w,   // Stream cho Systolic
    hls::stream<Tile4x4> &out_winograd_w,        // Stream cho Winograd
    hls::stream<int>& num_weight_vectors_stream                     // Tổng số vector cần điều phối
);

void data_demux(
    hls::stream<Tile4x4> &in_stream,     
    hls::stream<ap_uint<2>>& mode_stream,            
    hls::stream<Tile4x4> &out_systolic,  
    hls::stream<Tile4x4> &out_winograd,  
    hls::stream<int>& num_tiles_stream
);


#endif